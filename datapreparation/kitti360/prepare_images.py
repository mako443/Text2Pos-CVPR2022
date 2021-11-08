from typing import List

import os
import os.path as osp
import numpy as np
import pickle
import sys
import time

from shutil import copyfile

from scipy.spatial.distance import cdist

# import open3d

def sample_poses(path_poses, pose_distance):
    poses = np.loadtxt(path_poses)
    
    image_names = np.int0(poses[:, 0])

    orientations = poses[:, 1:].reshape((-1, 3, 4))
    orientations = orientations[:, 0:3, 0:3] # Take 3x3 matrices

    poses = poses[:, 1:].reshape((-1, 3,4)) # Convert to 3x4 matrices
    poses = poses[:, :, -1] # Take last column

    sampled_poses = [poses[0], ]
    sampled_orientations = [orientations[0], ]
    sampled_image_names = [image_names[0], ]

    for pose, orientation, image_name in zip(poses, orientations, image_names):
        dists = np.linalg.norm(pose - sampled_poses, axis=1)
        if np.min(dists) >= pose_distance:
            sampled_poses.append(pose)
            sampled_orientations.append(orientation)
            sampled_image_names.append(image_name)

    return np.array(sampled_poses), np.array(sampled_orientations), np.array(sampled_image_names)

def gather_images(path_images, ):
    pass

if __name__ == '__main__':
    np.random.seed(4096) # Set seed to re-produce results

    folder_name = '2013_05_28_drive_0010_sync' # Only on validation set

    path_poses = osp.join('./data/kitti360/data_poses', folder_name, 'poses.txt')
    path_images = osp.join('./data/kitti360-images/kitti360-data-2d', folder_name)
    pose_dist = 25 # In meters

    path_out = f'./data/k360_visloc_dist{pose_dist}/{folder_name}'

    if osp.isdir(path_out):
        quit('Output directory already exists!')
    
    poses, orientations, image_names = sample_poses(path_poses, pose_dist)
    print(f'Num poses: {len(poses)}')

    # Save / copy data
    os.mkdir(path_out)

    with open(osp.join(path_out, 'poses.pkl'), 'wb') as f:
        pickle.dump(poses, f)
    with open(osp.join(path_out, 'orientations.pkl'), 'wb') as f:
        pickle.dump(orientations, f)

    os.mkdir(osp.join(path_out, 'image_00'))
    for target_name, image_name in enumerate(image_names):
        path_src = osp.join(path_images, 'image_00', 'data_rect', f'{image_name:010.0f}.png')
        path_dst = osp.join(path_out, 'image_00', f'{target_name:04.0f}.png')
        copyfile(path_src, path_dst)
    print('DONE.')
            

    
             

