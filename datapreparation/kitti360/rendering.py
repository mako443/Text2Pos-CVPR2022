import numpy as np
import pickle
import os
import os.path as osp

from datapreparation.kitti360.imports import Object3d, Cell, Pose

import pptk
import open3d

def concat_objects(objects):
    xyz = np.vstack([o.xyz for o in objects])
    rgb = np.vstack([o.rgb for o in objects])
    return xyz, rgb

def render_poses(objects, poses, orientations):
    xyz, rgb = concat_objects(objects)
    viewer = pptk.viewer(xyz)
    viewer.attributes(rgb)

    _ = input() # Wait to set viewer to full-screen

    return viewer

if __name__ == '__main__':
    folder_name = '2013_05_28_drive_0010_sync'
    with open(osp.join('./data', 'kitti360', 'objects', f'{folder_name}.pkl'), 'rb') as f:
        objects = pickle.load(f)

    with open(osp.join('./data', 'k360_visloc_dist25', folder_name, 'poses.pkl'), 'rb') as f:
        poses = pickle.load(f)

    with open(osp.join('./data', 'k360_visloc_dist25', folder_name, 'orientations.pkl'), 'rb') as f:
        orientations = pickle.load(f)        

    xyz, rgb = concat_objects(objects)
    
    vis = open3d.visualization.Visualizer()
    vis.create_window(width=1408, height=376) # Same dimension as RGB pictures

    point_cloud=open3d.geometry.PointCloud()
    point_cloud.points=open3d.utility.Vector3dVector(xyz)
    point_cloud.colors=open3d.utility.Vector3dVector(rgb/255.0)

    vis.add_geometry(point_cloud)