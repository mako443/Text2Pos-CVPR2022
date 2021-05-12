from typing import List

import cv2
import os
import os.path as osp
import numpy as np
import pickle
import sys

import open3d
try:
    import pptk
except:
    print('pptk not found')
from plyfile import PlyData, PlyElement

from datapreparation.kitti360.drawing import show_pptk, show_objects, plot_cell
from datapreparation.kitti360.utils import CLASS_TO_LABEL, LABEL_TO_CLASS, COLORS, COLOR_NAMES, SCENE_NAMES
from datapreparation.kitti360.utils import CLASS_TO_MINPOINTS, CLASS_TO_VOXELSIZE, STUFF_CLASSES
from datapreparation.kitti360.imports import Object3d, Cell
from datapreparation.kitti360.descriptions import describe_cell

"""
DONE:
- Use closest point instead of center for description and plot?? Say 'on-top' if small distance => Seems good ✓

TODO:
- What about corrupted (?) scenes? Use bounding-boxes for instance objects?!
- How to handle multiple identical objects in matching? Remove from cell?
- Use "smarter" colors? E.g. top 1 or 2 histogram-buckets
"""

def load_points(filepath):
    plydata = PlyData.read(filepath)

    xyz = np.stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])).T
    rgb = np.stack((plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue'])).T

    lbl = plydata['vertex']['semantic']
    iid = plydata['vertex']['instance']

    return xyz, rgb, lbl, iid

def downsample_points(points, voxel_size):
    # voxel_size = 0.25
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(points.copy())
    _,_,indices_list = point_cloud.voxel_down_sample_and_trace(voxel_size,point_cloud.get_min_bound(), point_cloud.get_max_bound()) 
    # print(f'Downsampled from {len(points)} to {len(indices_list)} points')

    indices=np.array([ vec[0] for vec in indices_list ]) #Not vectorized but seems fast enough, CARE: first-index color sampling (not averaging)

    return indices

def extract_objects(xyz, rgb, lbl, iid):
    objects = []

    for label_name, label_idx in CLASS_TO_LABEL.items():
        mask = lbl == label_idx
        label_xyz, label_rgb, label_iid = xyz[mask], rgb[mask], iid[mask]

        for obj_iid in np.unique(label_iid):
            mask = label_iid == obj_iid
            obj_xyz, obj_rgb = label_xyz[mask], label_rgb[mask]

            obj_rgb = obj_rgb.astype(np.float32) / 255.0 # Scale colors [0,1]

            objects.append(Object3d(obj_xyz, obj_rgb, label_name, obj_iid))

    return objects
    
def gather_objects(path_input, folder_name):
    print(f'Loading objects for {folder_name}')

    path = osp.join(path_input, 'data_3d_semantics', folder_name, 'static')
    assert osp.isdir(path)
    file_names = [f for f in os.listdir(path) if not f.startswith('._')]

    scene_objects = {}

    for i_file_name, file_name in enumerate(file_names):
        # print(f'\t loading file {file_name}, {i_file_name} / {len(file_names)}')
        xyz, rgb, lbl, iid = load_points(osp.join(path, file_name))
        file_objects = extract_objects(xyz, rgb, lbl, iid)

        # Add new object or merge to existing
        merges = 0
        for obj in file_objects:
            if obj.id in scene_objects:
                scene_objects[obj.id] = Object3d.merge(scene_objects[obj.id], obj)
                merges += 1
            else:
                scene_objects[obj.id] = obj
            
            #Downsample the new or merged object
            voxel_size = CLASS_TO_VOXELSIZE[obj.label]
            if voxel_size is not None:
                indices = downsample_points(scene_objects[obj.id].xyz, voxel_size)
                scene_objects[obj.id].apply_downsampling(indices)
        # print(f'Merged {merges} / {len(file_objects)}')

    # Thresh objects by number of points
    objects = list(scene_objects.values())
    thresh_counts = {}
    objects_threshed = []
    for obj in objects:
        if len(obj.xyz) < CLASS_TO_MINPOINTS[obj.label]:
            if obj.label in thresh_counts:
                thresh_counts[obj.label] += 1
            else:
                thresh_counts[obj.label] = 1
        else:
            objects_threshed.append(obj)
    print(thresh_counts)

    return objects_threshed
    # return list(scene_objects.values())

def get_close_poses(poses: List[np.ndarray], scene_objects: List[Object3d], cell_size, pose_objects=None):
    """Retains all poses that are at most cell_size / 2 distant from an instance-object.

    Args:
        poses (List[np.ndarray]): [description]
        scene_objects (List[Object3d]): [description]
        cell_size ([type]): [description]
        pose_objects ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    instance_objects = [obj for obj in scene_objects if obj.label not in STUFF_CLASSES]
    close_poses, close_pose_objects = [], []
    for i_pose, pose in enumerate(poses):
        for obj in instance_objects:
            closest_point = obj.get_closest_point(pose)
            dist = np.linalg.norm(pose - closest_point)
            obj.closest_point = None
            if dist < cell_size / 2:
                close_poses.append(pose)
                close_pose_objects.append(pose_objects[i_pose])
                break
    
    assert len(close_poses) > len(poses) * 2/5, f'Too few poses retained ({len(close_poses)} of {len(poses)}), are all objects loaded?'
    print(f'closest poses: {len(close_poses)} of {len(poses)}')

    if pose_objects:
        return close_poses, close_pose_objects
    else:
        return close_poses

def create_poses(path_input, path_output, folder_name, pose_distance, return_pose_objects=False):
    path = osp.join(path_input, 'data_poses', folder_name, 'poses.txt')
    poses = np.loadtxt(path)
    poses = poses[:, 1:].reshape((-1, 3,4)) # Convert to 3x4 matrices
    poses = poses[:, :, -1] # Take last column

    # CARE: This can still lead to two very close-by poses if the trajectory "went around a corner"
    sampled_poses = [poses[0], ]
    for pose in poses:
        # dist = np.linalg.norm(pose - sampled_poses[-1])
        # if dist >= pose_distance:
        #     sampled_poses.append(pose)
        dists = np.linalg.norm(pose - sampled_poses, axis=1)
        if np.min(dists) >= pose_distance:
            sampled_poses.append(pose)

    if return_pose_objects:
        pose_objects = []
        for pose in sampled_poses:
            pose_objects.append(Object3d(
                np.random.rand(50, 3)*3 + pose,
                np.ones((50, 3)),
                '_pose',
                99
            ))
        print(f'{folder_name} sampled {len(sampled_poses)} poses')
        return sampled_poses, pose_objects
    else:
        return sampled_poses

def create_cells(objects, poses, scene_name, cell_size):
    print('Creating cells...')
    cells = []
    none_indices = []
    for i_pose, pose in enumerate(poses):
        # print(f'\r \t pose {i_pose} / {len(poses)}', end='')
        bbox = np.hstack((pose - cell_size/2, pose + cell_size/2)) # [x0, y0, z0, x1, y1, z1]

        # Shift the cell in x-y-plane
        shift = np.random.randint(-cell_size//2.2, cell_size//2.2, size=2) # Shift so that pose is guaranteed to be in cell
        bbox[0:2] += shift
        bbox[3:5] += shift

        cell = describe_cell(bbox, objects, pose, scene_name)
        if cell is not None:
            cells.append(cell)
        else:
            # print(f'\n None at {i_pose}\n')
            none_indices.append(i_pose)
    # print()
    
    print(f'Nones: {len(none_indices)} / {len(poses)}')
    if len(none_indices) > len(poses)/3:
        print(f'Too many nones, are all objects gathered?')
        return False, none_indices
    else:
        return True, cells

    return cells

if __name__ == '__main__':
    np.random.seed(4096) # Set seed to re-produce results
    path_input = './data/kitti360'
    path_output = './data/kitti360_overlap_1_2'
    scene_name = sys.argv[-1]
    print('Scene:', scene_name)
    scene_names = SCENE_NAMES if scene_name=='all' else [scene_name, ]

    cell_size = 30
    print(f'Preparing {scene_names} {path_input} -> {path_output}, cell_size {cell_size}')

    # Incomplete folders: 3 corrupted...
    # for folder_name in SCENE_NAMES:
    for folder_name in scene_names: # 2013_05_28_drive_0000_sync
        print(f'Folder: {folder_name}')

        # poses, pose_objects = create_poses(path_input, path_output, folder_name, cell_size, return_pose_objects=True)        
        poses, pose_objects = create_poses(path_input, path_output, folder_name, pose_distance=cell_size * 1 / 2, return_pose_objects=True)        

        path_objects = osp.join(path_output, 'objects', f'{folder_name}.pkl')
        path_cells = osp.join(path_output, 'cells', f'{folder_name}.pkl')

        # Load or gather objects
        if not osp.isfile(path_objects): # Build if not cached
            objects = gather_objects(path_input, folder_name)
            pickle.dump(objects, open(path_objects, 'wb'))
            print(f'Saved objects to {path_objects}')  
        else:
            print(f'Loaded objects from {path_objects}')
            objects = pickle.load(open(path_objects, 'rb'))

        poses, pose_objects = get_close_poses(poses, objects, cell_size, pose_objects)

        # show_objects(objects + pose_objects)
        # quit()

        # Create cells
        res, cells = create_cells(objects, poses, folder_name, cell_size)
        assert res is True, "Too many nones, quitting."

        pickle.dump(cells, open(path_cells, 'wb'))
        print(f'Saved {len(cells)} cells to {path_cells}')   

        # Debugging 
        idx = np.random.randint(len(cells))
        idx = np.random.randint(len(cells))
        cell = cells[idx]
        print('idx', idx)
        print(cell.get_text())
        print(cell.bbox_w)

        img = plot_cell(cell)
        cv2.imwrite(f'cell_demo_idx{idx}.png', img)

        # except:
        #     print(f'Scene {folder_name} failed, removing objects again')
        #     os.remove(path_objects)
        
        print('--- \n')


    # show_objects(cell.objects, scale=100)

    # viewer = show_objects(objects + pose_objects)
    # lens = [len(o.xyz) for o in objects]
    # print(f'{len(objects)} objects from {folder_name} w/ {np.mean(lens):0.0f} avg. points')