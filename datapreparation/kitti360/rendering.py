import numpy as np
import pickle
import os
import os.path as osp

from scipy.spatial.transform import Rotation

from datapreparation.kitti360.imports import Object3d, Cell, Pose

import pptk
import cv2

def concat_objects(objects):
    xyz = np.vstack([o.xyz for o in objects])
    rgb = np.vstack([o.rgb for o in objects])
    return xyz, rgb

def create_viewer(objects):
    xyz, rgb = concat_objects(objects)
    viewer = pptk.viewer(xyz)
    viewer.attributes(rgb)
    viewer.set(point_size=0.1)
    return viewer

def get_orientations_manually(objects, poses, step=3):
    viewer = create_viewer(objects)

    indices = []
    values = []

    for idx in range(0, len(poses), step):
        viewer.set(lookat=poses[idx])
        print(f'Idx: {idx}')
        
        if idx == 108:
            continue

        img = cv2.imread(f'./data/k360_visloc_dist25/2013_05_28_drive_0010_sync/image_00/{idx:04.0f}.png')
        cv2.imshow("wdw",img)
        cv2.waitKey(1)

        viewer.wait()
        
        indices.append(idx)
        values.append(viewer.get('phi'))
        print(f'\t Saved {values[-1]}\n')

        with open('vals.txt', 'a') as f:
            f.write(f'{idx} {values[-1]}\n')

    orientations_interpolated = np.interp(np.arange(len(poses)), indices, values)
    with open('orientations_interpolated.pkl', 'wb') as f:
        pickle.dump(orientations_interpolated, f)
    print(orientations_interpolated.reshape((-1, 1))) 

def set_angle(idx, offset=np.pi):
    forward = poses[idx+1] - poses[idx]
    angle = np.arctan2(forward[1], forward[0]) + offset
    v.set(phi=angle, lookat=poses[idx])    



if __name__ == '__main__':
    folder_name = '2013_05_28_drive_0010_sync'
    with open(osp.join('./data', 'kitti360', 'objects', f'{folder_name}.pkl'), 'rb') as f:
        objects = pickle.load(f)

    with open(osp.join('./data', 'k360_visloc_dist25', folder_name, 'poses.pkl'), 'rb') as f:
        poses = pickle.load(f)

    with open(osp.join('./data', 'k360_visloc_dist25', folder_name, 'orientations.pkl'), 'rb') as f:
        orientations = pickle.load(f)        

    v = create_viewer(objects)
    
    # vis = open3d.visualization.Visualizer()
    # vis.create_window(width=1408, height=376) # Same dimension as RGB pictures

    # point_cloud=open3d.geometry.PointCloud()
    # point_cloud.points=open3d.utility.Vector3dVector(xyz)
    # point_cloud.colors=open3d.utility.Vector3dVector(rgb/255.0)

    # vis.add_geometry(point_cloud)

    # v = render_poses(objects, None, None)
    # v.set(lookat=poses[0])

    # cam0 = np.loadtxt('./data/kitti360/data_poses/2013_05_28_drive_0010_sync/cam0_to_world.txt')
    # cam0 = cam0[0, 1:].reshape(4,4).astype(np.float16)
    # rot = cam0[0:3, 0:3]