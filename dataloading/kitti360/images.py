from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2
from copy import deepcopy

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

class Kitti360ImageDataset(Dataset):
    def __init__(self, base_path, scene_name, db_or_query, query_fraction=4, transform=None):
        assert db_or_query in ('db', 'query')

        self.base_path = base_path
        self.scene_name = scene_name
        self.transform = transform

        with open(osp.join(base_path, scene_name, 'poses.pkl'), 'rb') as f:
            self.poses = pickle.load(f)

        with open(osp.join(base_path, scene_name, 'orientations.pkl'), 'rb') as f:
            self.orientations = pickle.load(f)

        self.image_paths = np.array([
            osp.join(base_path, scene_name, 'image_00', f) for f in os.listdir(osp.join(base_path, scene_name, 'image_00')) if f.endswith('.png')
        ])
        
        mask = np.array([True for _ in range(len(self.poses))])
        mask[: : query_fraction] = False
        if db_or_query == 'query':
            mask = np.invert(mask)

        self.poses, self.orientations, self.image_paths = self.poses[mask], self.orientations[mask], self.image_paths[mask]

        assert len(self.poses) == len(self.orientations) == len(self.image_paths), 'Number of poses, orientations and images does not match.'

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])

        if self.transform:
            img = self.transform(img)

        return {
            'poses': self.poses[idx],
            'orientations': self.orientations[idx],
            'images': img
        }

if __name__ == '__main__':
    dataset = Kitti360ImageDataset('./data/k360_visloc_dist25', '2013_05_28_drive_0010_sync', 'db')
    data = dataset[0]
    print(data)