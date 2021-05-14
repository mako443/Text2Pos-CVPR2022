from typing import List

import cv2
import os
import os.path as osp
import numpy as np
import pickle
import sys
import time

import open3d
try:
    import pptk
except:
    print('pptk not found')
from plyfile import PlyData, PlyElement

from datapreparation.kitti360.drawing import show_pptk, show_objects, plot_cell, plot_pose_in_best_cell
from datapreparation.kitti360.utils import CLASS_TO_LABEL, LABEL_TO_CLASS, COLORS, COLOR_NAMES, SCENE_NAMES
from datapreparation.kitti360.utils import CLASS_TO_MINPOINTS, CLASS_TO_VOXELSIZE, STUFF_CLASSES
from datapreparation.kitti360.imports import Object3d, Cell, Pose
from datapreparation.kitti360.descriptions import create_cell, describe_pose_in_pose_cell, describe_pose_in_best_cell

"""
DONE:
- Use closest point instead of center for description and plot?? Say 'on-top' if small distance => Seems good ✓

TODO:
- Say "inside" or something for buildings (instead of "on-top")
- Set some color names -.-
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

            # objects.append(Object3d(obj_xyz, obj_rgb, label_name, obj_iid))
            objects.append(Object3d(obj_iid, obj_iid, obj_xyz, obj_rgb, label_name)) # Initially also set id instance-id for later mergin. Re-set in create_cell()

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

def get_close_locations(locations: List[np.ndarray], scene_objects: List[Object3d], cell_size, location_objects=None):
    """Retains all locations that are at most cell_size / 2 distant from an instance-object.

    Args:
        locations (List[np.ndarray]): [description]
        scene_objects (List[Object3d]): [description]
        cell_size ([type]): [description]
        location_objects ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    instance_objects = [obj for obj in scene_objects if obj.label not in STUFF_CLASSES]
    close_locations, close_location_objects = [], []
    for i_location, location in enumerate(locations):
        for obj in instance_objects:
            closest_point = obj.get_closest_point(location)
            dist = np.linalg.norm(location - closest_point)
            obj.closest_point = None
            if dist < cell_size / 2:
                close_locations.append(location)
                close_location_objects.append(location_objects[i_location])
                break
    
    assert len(close_locations) > len(locations) * 2/5, f'Too few locations retained ({len(close_locations)} of {len(locations)}), are all objects loaded?'
    print(f'close locations: {len(close_locations)} of {len(locations)}')

    if location_objects:
        return close_locations, close_location_objects
    else:
        return close_locations

def create_locations(path_input, path_output, folder_name, pose_distance, return_location_objects=False):
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

    if return_location_objects:
        pose_objects = []
        for pose in sampled_poses:
            pose_objects.append(Object3d(-1, -1, 
                np.random.rand(50, 3)*3 + pose,
                np.ones((50, 3)),
                '_pose'
            ))
        print(f'{folder_name} sampled {len(sampled_poses)} poses')
        return sampled_poses, pose_objects
    else:
        return sampled_poses

def create_cells(objects, locations, scene_name, cell_size) -> List[Cell]:
    print('Creating cells...')
    print('CARE: Not shifting cells!')
    cells = []
    none_indices = []
    
    assert len(scene_name.split('_')) == 6
    scene_name_short = scene_name.split('_')[-2]

    for i_location, location in enumerate(locations):
        # print(f'\r \t locations {i_location} / {len(locations)}', end='')
        bbox = np.hstack((location - cell_size/2, location + cell_size/2)) # [x0, y0, z0, x1, y1, z1]

        # Shift the cell in x-y-plane
        # shift = np.random.randint(-cell_size//2.2, cell_size//2.2, size=2) # Shift so that pose is guaranteed to be in cell
        # bbox[0:2] += shift
        # bbox[3:5] += shift

        cell = create_cell(i_location, scene_name_short, bbox, objects)
        if cell is not None:
            cells.append(cell)
        else:
            none_indices.append(i_location)
    
    print(f'None cells: {len(none_indices)} / {len(locations)}')
    if len(none_indices) > len(locations)/3:
        return False, none_indices
    else:
        return True, cells

# TODO: Simpler formulation with deleting objects? How would I gather the objects for describing?
def create_poses(objects: List[Object3d], locations, cells: List[Cell], scene_name) -> List[Pose]:
    """[summary]
    Create cells -> sample pose location -> describe with pose-cell -> convert description to best-cell for training

    Args:
        objects (List[Object3d]): [description]
        locations ([type]): [description]
        cells (List[Cell]): [description]
        scene_name ([type]): [description]

    Returns:
        [type]: [description]
    """
    print('Creating poses...')
    poses = []
    none_indices = []

    cell_centers = np.array([cell.bbox_w for cell in cells])
    cell_centers = 1/2 * (cell_centers[:, 0:3] + cell_centers[:, 3:6])
    cell_size = cells[0].cell_size

    unmatched_counts = []
    for i_location, location in enumerate(locations):
        # Find closest cell
        dists = np.linalg.norm(location - cell_centers, axis=1)
        best_cell = cells[np.argmin(dists)]

        if np.min(dists) > cell_size / 2:
            none_indices.append(i_location)
            continue

        # Create an extra cell on top of the pose to create the query-side description decoupled from the database-side cells.
        pose_cell_bbox = np.hstack((location - cell_size/2, location + cell_size/2)) # [x0, y0, z0, x1, y1, z1]
        pose_cell = create_cell(-1, "pose", pose_cell_bbox, objects)
        assert pose_cell is not None, f'{pose_cell_bbox}, {location}' #TODO: Can be None after shifting, then discard this shifted pose

        # Obtain the descriptions based on the pose-cell
        descriptions = describe_pose_in_pose_cell(location, pose_cell)

        # Convert the descriptions to the best-matching database cell for training. Some descriptions might not be matched anymore.
        descriptions, pose_in_cell, num_unmatched = describe_pose_in_best_cell(location, descriptions, best_cell)
        unmatched_counts.append(num_unmatched)

        pose = Pose(pose_in_cell, location, best_cell.id, descriptions)
        poses.append(pose)

    print(f'None poses: {len(none_indices)} / {len(locations)}, avg. unmatched: {np.mean(unmatched_counts):0.1f}')
    if len(none_indices) > len(locations)/3:
        return False, none_indices
    else:
        return True, poses          

if __name__ == '__main__':
    np.random.seed(4096) # Set seed to re-produce results
    path_input = './data/kitti360'
    path_output = './data/k360_decouple'
    scene_name = sys.argv[-1]
    print('Scene:', scene_name)
    scene_names = SCENE_NAMES if scene_name=='all' else [scene_name, ]

    cell_size = 30
    print(f'Preparing {scene_names} {path_input} -> {path_output}, cell_size {cell_size}')

    for folder_name in scene_names: # 2013_05_28_drive_0000_sync
        print(f'Folder: {folder_name}')

        locations, location_objects = create_locations(path_input, path_output, folder_name, pose_distance=cell_size, return_location_objects=True)        

        path_objects = osp.join(path_output, 'objects', f'{folder_name}.pkl')
        path_cells = osp.join(path_output, 'cells', f'{folder_name}.pkl')
        path_poses = osp.join(path_output, 'poses', f'{folder_name}.pkl')

        t_start = time.time()        

        # Load or gather objects
        if not osp.isfile(path_objects): # Build if not cached
            objects = gather_objects(path_input, folder_name)
            pickle.dump(objects, open(path_objects, 'wb'))
            print(f'Saved objects to {path_objects}')  
        else:
            print(f'Loaded objects from {path_objects}')
            objects = pickle.load(open(path_objects, 'rb'))

        t_object_loaded = time.time()

        locations, location_objects = get_close_locations(locations, objects, cell_size, location_objects)

        t_close_locations = time.time()

        location = locations[0]
        res, cells = create_cells(objects, locations, folder_name, cell_size)
        assert res is True, "Too many cell nones, quitting."

        t_cells_created = time.time()

        res, poses = create_poses(objects, locations, cells, folder_name)
        assert res is True, "Too many pose nones, quitting."

        t_poses_created = time.time()

        print('Ela objects', t_object_loaded - t_start)
        print('Ela close', t_close_locations - t_object_loaded)
        print('Ela cells', t_cells_created - t_close_locations)
        print('Ela poses', t_poses_created - t_cells_created)

        pickle.dump(cells, open(path_cells, 'wb'))
        print(f'Saved {len(cells)} cells to {path_cells}')   

        pickle.dump(poses, open(path_poses, 'wb'))
        print(f'Saved {len(poses)} poses to {path_poses}')           


        # Debugging 
        idx = np.random.randint(len(poses))
        pose = poses[idx]
        cells_dict = {cell.id: cell for cell in cells}
        cell = cells_dict[pose.cell_id]
        print('IDX:', idx)
        print(pose.get_text())

        img = plot_pose_in_best_cell(cell, pose)
        cv2.imwrite(f'cell_demo_idx{idx}.png', img)
