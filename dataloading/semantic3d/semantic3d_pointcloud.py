import os
import os.path as osp
import sys
import time
import h5py
import json
import pickle
import numpy as np

import random
import cv2

import torch
from torch.utils.data import Dataset

import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader

from datapreparation.semantic3d.imports import Object3D, DescriptionObject, COMBINED_SCENE_NAMES, COLORS, COLOR_NAMES
from datapreparation.semantic3d.drawing import draw_cells

class Semantic3dObjectDatasetMulti(Dataset):
    def __init__(self, path_numpy, path_scenes, scene_names, split=None, transform=T.Compose([T.FixedPoints(2048), T.NormalizeScale()])):
        self.path_numpy = path_numpy
        self.path_scenes = path_scenes
        self.split = split
        self.scene_names = scene_names

        self.datasets = [Semantic3dObjectDataset(path_numpy, path_scenes, scene_name, split=split, transform=transform) for scene_name in self.scene_names]

        self.known_classes = self.datasets[0].known_classes

    def __len__(self):
        return np.sum([len(ds) for ds in self.datasets])

    def __getitem__(self, idx):
        count = 0
        for dataset in self.datasets:
            idx_in_dataset = idx - count
            if idx_in_dataset < len(dataset):
                return dataset[idx_in_dataset]
            else:
                count += len(dataset)
        assert False

class Semantic3dObjectDataset(Dataset):
    def __init__(self, path_numpy, path_scenes, scene_name, split=None, transform=T.Compose([T.FixedPoints(2048), T.NormalizeScale()])):
        self.path_numpy = path_numpy
        self.path_scenes = path_scenes
        self.split = split  
        self.scene_name = scene_name

        #Load objects
        self.scene_objects = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'objects.pkl'), 'rb')) 
        #Possibly apply split
        if split is not None:
            assert split in ('train', 'test')
            test_indices = (np.arange(len(self.scene_objects)) % 5) == 0
            indices = test_indices if split=='test' else np.bitwise_not(test_indices)
            self.scene_objects = [obj for (idx, obj) in enumerate(self.scene_objects) if indices[idx]]

        random.shuffle(self.scene_objects) # Prevent objects from being in order by classes

        self.class_to_index = {'unlabeled': 0, 'man-made terrain': 1, 'natural terrain': 2, 'high vegetation': 3, 'low vegetation': 4, 'buildings': 5, 'hard scape': 6, 'scanning artefacts': 7, 'cars': 8}
        self.known_classes = list(self.class_to_index.keys())

        # self.known_classes = list(np.unique([obj.label for obj in self.scene_objects]))
        # self.class_to_index = {c: i for (i,c) in enumerate(self.known_classes)}

        # self.transform = T.Compose([T.NormalizeScale(), T.FixedPoints(num_points)])
        self.transform = transform

        print(f'Semantic3dObjectDataset ({self.scene_name}): {len(self)} objects')

    def __len__(self):
        return len(self.scene_objects)

    def __getitem__(self, idx):
        obj = self.scene_objects[idx]
        points = torch.tensor(np.float32(obj.points_w))
        colors = torch.tensor(np.float32(obj.points_w_color))
        label = self.class_to_index[obj.label]
        
        data = Data(x=colors, y=label, pos=points)

        data = self.transform(data)
        return data

if __name__ == '__main__':
    dataset = Semantic3dObjectDataset('./data/numpy_merged/', './data/semantic3d')
    dataloader = DataLoader(dataset, batch_size=2)
    data = dataset[0]
    batch = next(iter(dataloader))