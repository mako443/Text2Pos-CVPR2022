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

from datapreparation.imports import Object3D, DescriptionObject, COMBINED_SCENE_NAMES, COLORS, COLOR_NAMES
from datapreparation.drawing import draw_cells

class Semantic3dObjectDataset(Dataset):
    def __init__(self, path_numpy, path_scenes, split=None, num_points=2048):
        self.path_numpy = path_numpy
        self.path_scenes = path_scenes
        self.split = split  

        self.scene_name = 'neugasse_station1_xyz_intensity_rgb'

        #Load objects
        self.scene_objects = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'objects.pkl'), 'rb')) 
        #Possibly apply split
        if split is not None:
            assert split in ('train', 'test')
            test_indices = (np.arange(len(self.scene_objects)) % 5) == 0
            indices = test_indices if split=='test' else np.bitwise_not(test_indices)
            self.scene_objects = [obj for (idx, obj) in enumerate(self.scene_objects) if indices[idx]]

        random.shuffle(self.scene_objects) # Prevent objects from being in order by classes

        self.known_classes = list(np.unique([obj.label for obj in self.scene_objects]))
        self.class_to_index = {c: i for (i,c) in enumerate(self.known_classes)}

        self.transform = T.Compose([T.NormalizeScale(), T.FixedPoints(num_points)])

        print(f'Semantic3dObjectDataset: {len(self)} objects, using {num_points} points')

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