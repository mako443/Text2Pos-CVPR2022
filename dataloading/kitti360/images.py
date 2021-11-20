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

class Kitti360ImageCompareDataset(Dataset):
    def __init__(self, base_path, scene_name, db_or_query, transform=None):
        assert db_or_query in ('db', 'query')

        self.base_path = base_path
        self.scene_name = scene_name
        self.transform = transform

        with open(osp.join(base_path, 'visloc', scene_name, db_or_query, 'poses.pkl'), 'rb') as f:
            self.poses = pickle.load(f)

        image_dirpath = osp.join(base_path, 'visloc', scene_name, db_or_query)
        self.image_paths = sorted([osp.join(image_dirpath, f) for f in os.listdir(image_dirpath) if f.endswith('.png')])
        self.image_paths = np.array(self.image_paths)
        assert len(self.poses) == len(self.image_paths)

        print(f'Kitti360ImageCompareDataset: {len(self)} poses for {db_or_query}.')

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])

        if self.transform:
            img = self.transform(img)

        return {
            'poses': self.poses[idx],
            'images': img
        }    

class Kitti360ImageDataset(Dataset):
    def __init__(self, base_path, scene_name, split, transform=None):
        assert split in ('db', 'query')

        self.base_path = base_path
        self.scene_name = scene_name
        self.split = split
        self.transform = transform

        with open(osp.join(base_path, scene_name, f'poses_{split}.pkl'), 'rb') as f:
            self.poses = pickle.load(f)

        # NOTE: File names have to be sorted!
        self.image_paths = np.array(sorted([
            osp.join(base_path, scene_name, 'real', split, f) for f in os.listdir(osp.join(base_path, scene_name, 'real', split)) if f.endswith('.png')
        ]))

        if osp.isdir(osp.join(base_path, scene_name, 'rendered', split)):
            self.image_paths_rendered = np.array(sorted([
                osp.join(base_path, scene_name, 'rendered', split, f) for f in os.listdir(osp.join(base_path, scene_name, 'rendered', split)) if f.endswith('.png')
            ]))
        else:
            self.image_paths_rendered = None
        

        assert len(self.poses) == len(self.image_paths), 'Number of poses and images do not match.'
        if self.image_paths_rendered is not None:
            assert len(self.image_paths) == len(self.image_paths_rendered)

        print(f'Kitti360ImageDataset ({self.scene_name}): {len(self)} poses, rendered: {self.image_paths_rendered is not None}')

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        if self.image_paths_rendered is not None:
            img_rendered = Image.open(self.image_paths_rendered[idx])
        else:
            img_rendered = None

        if self.transform:
            img = self.transform(img)
            if img_rendered is not None:
                img_rendered = self.transform(img_rendered)

        data = {
            'poses': self.poses[idx],
            'images': img,
        }
        if img_rendered is not None:
            data['images_rendered'] = img_rendered

        return data

if __name__ == '__main__':
    dataset = Kitti360ImageCompareDataset('./data/k360_30-10_scG_pd10_pc4_spY_all', '2013_05_28_drive_0010_sync', 'db')
    data = dataset[0]

    #dataset = Kitti360ImageDataset('./data/k360-visloc_db-25_q5', '2013_05_28_drive_0010_sync', 'db')
    #data = dataset[0]
    #print(data)
