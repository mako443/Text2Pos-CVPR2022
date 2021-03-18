import cv2
import os
import os.path as osp
import numpy as np

import open3d
import pptk
from plyfile import PlyData, PlyElement

from datapreparation.kitti360.drawing import show_pptk, show_objects
from datapreparation.kitti360.utils import CLASS_TO_LABEL, LABEL_TO_CLASS, CLASS_TO_MINPOINTS
from datapreparation.kitti360.imports import Object3d

"""
TODO:
- per-class voxel-sizes
- expressive enough? do non-id objects have bounding boxes??
"""

def load_points(filepath):
    plydata = PlyData.read(filepath)

    xyz = np.stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])).T
    rgb = np.stack((plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue'])).T

    lbl = plydata['vertex']['semantic']
    iid = plydata['vertex']['instance']

    return xyz, rgb, lbl, iid

def downsample_points(points):
    voxel_size = 0.5
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

            # retained_indices = downsample_points(obj_xyz)
            # obj_xyz, obj_rgb = obj_xyz[retained_indices], obj_rgb[retained_indices]

            objects.append(Object3d(obj_xyz, obj_rgb, label_name, obj_iid))

    return objects
    
def load_scene(base_path, folder_name):
    path = osp.join(base_path, 'data_3d_semantics', folder_name, 'static')
    file_names = [f for f in os.listdir(path) if not f.startswith('._')]

    scene_objects = {}

    for i_file_name, file_name in enumerate(file_names):
        print(f'\t loading file {file_name}, {i_file_name} / {len(file_names)}')
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
            indices = downsample_points(scene_objects[obj.id].xyz)
            scene_objects[obj.id].apply_downsampling(indices)
        # print(f'Merged {merges} / {len(file_objects)}')

    return list(scene_objects.values())
    
if __name__ == '__main__':
    base_path = './data/kitti360'
    folder_name = '2013_05_28_drive_0000_sync'

    objects = load_scene(base_path, folder_name)
    viewer = show_objects(objects)
    