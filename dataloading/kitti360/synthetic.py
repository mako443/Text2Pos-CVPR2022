from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2
from easydict import EasyDict
import time

import torch
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T     

from datapreparation.kitti360.utils import CLASS_TO_LABEL, LABEL_TO_CLASS, CLASS_TO_MINPOINTS, CLASS_TO_INDEX
from datapreparation.kitti360.utils import COLORS, COLOR_NAMES, SCENE_NAMES_TRAIN
from datapreparation.kitti360.descriptions import create_cell, describe_pose
from datapreparation.kitti360.imports import Object3d, Cell
from datapreparation.kitti360.drawing import show_pptk, show_objects, plot_cell, plot_pose
from dataloading.kitti360.base import Kitti360BaseDataset
from dataloading.kitti360.objects import Kitti360ObjectsDatasetMulti
from dataloading.kitti360.poses import load_pose_and_cell


'''
TODO:
- Sample class first or not?
- Explicitly move an object to on-top first or not?
'''

class Kitti360FineSyntheticDataset(Dataset):
    def __init__(self, base_path, scene_names, transform, args, length=1024, fixed_seed=False):
        # Create an objects dataset to copy the objects from in synthetic cell creation
        # CARE: some classes might be empty because of the train/test split
        objects_dataset = Kitti360ObjectsDatasetMulti(base_path, scene_names) # Transform of this dataset is ignored
        self.objects_dict = {c: [] for c in CLASS_TO_INDEX.keys()}
        for obj in objects_dataset.objects:
            self.objects_dict[obj.label].append(obj)

        self.transform = transform
        self.pad_size = args.pad_size
        self.num_mentioned = args.num_mentioned
        self.length = length
        self.fixed_seed = fixed_seed
        self.colors = COLORS
        self.color_names = COLOR_NAMES 

        print(f'Kitti360FineSyntheticDataset, fixed seed: {fixed_seed}, length: {length}, sampling from {len(objects_dataset)} objects')    

    def create_synthetic_cell_and_pose(self):
        pose = np.random.rand(3)
        num_distractors = np.random.randint(self.pad_size - self.num_mentioned) if self.pad_size > self.num_mentioned else 0

        # Copy over random objects from the real dataset to random positions
        # Note that the objects are already clustered and normed: taken from cells (not scene) in Kitti360ObjectsDataset
        cell_objects = []
        for i in range(self.num_mentioned + num_distractors):
            obj_class = np.random.choice([k for k, v in self.objects_dict.items() if len(v) > 0])
            obj = np.random.choice(self.objects_dict[obj_class])

            # Shift the object center to a random position ∈ [0, 1] in x-y-plane, z is kept
            # Note that object might be partly outside the cell, but that is ok when masking + clustering is skipped
            obj.xyz[:, 0:2] -= np.mean(obj.xyz[:, 0:2], axis=0)
            obj.xyz[:, 0:2] += np.random.rand(2)

            cell_objects.append(obj)
            # TODO: possibly apply transforms

        # Randomly shift an object close to the pose for <on-top>
        if np.random.choice((True, False)):
            idx = np.random.randint(len(cell_objects))
            obj = cell_objects[idx]
            obj.xyz[:, 0:2] -= np.mean(obj.xyz[:, 0:2], axis=0)
            obj.xyz[:, 0:2] += np.array(pose[0:2] + np.random.randn(2)*0.01).reshape((1,2))

        # Create the cell
        cell = create_cell(-1, "MOCK", np.array([0,0,0,1,1,1]), cell_objects, is_synthetic=True)
        assert cell is not None
        assert np.allclose(cell.cell_size, 1.0)  

        # Create the pose
        pose = describe_pose(pose, cell)

        return cell, pose

    def __getitem__(self, idx):
        """Return the data of a synthetic cell.
        """
        if self.fixed_seed:
            np.random.seed(idx)
        t0 = time.time()

        cell, pose = self.create_synthetic_cell_and_pose()
        hints = Kitti360BaseDataset.create_hint_description(pose, cell)

        return load_pose_and_cell(pose, cell, hints, self.pad_size, self.transform)

    def __len__(self):
        return self.length

    def collate_fn(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch      

    def get_known_classes(self):
        return list(self.objects_dict.keys())

    def get_known_words(self):
        known_words = []
        for i in range(50):
            data = self[i]
            for hint in data['hint_descriptions']:
                known_words.extend(hint.replace('.','').replace(',','').lower().split())
        return list(np.unique(known_words))  

if __name__ == '__main__':
    base_path = './data/k360_decouple'
    folder_name = '2013_05_28_drive_0003_sync'
    args = EasyDict(pad_size=8, num_mentioned=6)

    transform = T.Compose([T.FixedPoints(1024), T.NormalizeScale(), T.RandomFlip(0), T.RandomFlip(1), T.RandomFlip(2), T.NormalizeScale()])
    dataset = Kitti360FineSyntheticDataset(base_path, [folder_name, ], transform, args)
    data = dataset[0]
    
