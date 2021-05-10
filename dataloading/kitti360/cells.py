from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader

import torch_geometric.transforms as T 

from datapreparation.kitti360.utils import CLASS_TO_LABEL, LABEL_TO_CLASS, CLASS_TO_MINPOINTS, SCENE_NAMES
from datapreparation.kitti360.utils import CLASS_TO_INDEX, COLOR_NAMES
from datapreparation.kitti360.imports import Object3d, Cell
from datapreparation.kitti360.drawing import show_pptk, show_objects, plot_cell
from dataloading.kitti360.base import Kitti360BaseDataset
from dataloading.kitti360.poses import batch_object_points

'''
Augmentations:
- hints order (care not to influence matches)
- pads to random objects and vice-versa
- flip cell
'''
class Kitti360CellDataset(Kitti360BaseDataset):
    def __init__(self, base_path, scene_name, transform, split=None, shuffle_hints=False, flip_cells=False):
        super().__init__(base_path, scene_name, split)
        self.shuffle_hints = shuffle_hints
        self.transform = transform
        self.flip_cells = flip_cells

    def __getitem__(self, idx):
        cell = self.cells[idx]
        hints = self.hint_descriptions[idx]
        
        if self.shuffle_hints:
            hints = np.random.choice(hints, size=len(hints), replace=False)

        text = ' '.join(hints)

        # CARE: hints are currently not flipped!
        if self.flip_cells:
            if np.random.choice((True, False)): # Horizontal
                cell, text = flip_cell(cell, text, 1)
            if np.random.choice((True, False)): # Vertical
                cell, text = flip_cell(cell, text, -1)                
                

        object_points = batch_object_points(cell.objects, self.transform)

        object_class_indices = [CLASS_TO_INDEX[obj.label] for obj in cell.objects]
        object_color_indices = [COLOR_NAMES.index(obj.get_color_text()) for obj in cell.objects]           

        return {
            'cells': cell,
            'objects': cell.objects,
            'object_points': object_points,
            'texts': text,
            # 'cell_indices': idx,
            'scene_names': self.scene_name,
            'object_class_indices': object_class_indices,
            'object_color_indices': object_color_indices            
        } 

    def __len__(self):
        return len(self.cells)

# TODO: for free orientations, possibly flip cell only, create descriptions and hints again
# OR: numeric vectors in descriptions, flip cell objects and description.direction, then create hints again
# Flip pose, too?
def flip_cell(cell, text, direction):
    """Flips the cell horizontally or vertically
    CARE: Needs adjustment for non-compass directions

    Args:
        cell (Cell): The cell to flip, is copied before modification
        text (str): The text description to flip
        direction (int): Horizontally (+1) or vertically (-1)

    Returns:
        Cell: flipped cell
        str: flipped text
    """
    assert direction in (-1, 1)

    cell = deepcopy(cell)

    if direction == 1: #Horizontally
        for obj in cell.objects:
            obj.xyz[:, 0] = 1 - obj.xyz[:, 0]
            obj.closest_point[0] = 1 - obj.closest_point[0]

        text = text.replace('east','east-flipped').replace('west','east').replace('east-flipped', 'west')
    elif direction == -1: #Vertically
        for obj in cell.objects:
            obj.xyz[:, 1] = 1 - obj.xyz[:, 1]
            obj.closest_point[1] = 1 - obj.closest_point[1]  
              
        text = text.replace('north', 'north-flipped'). replace('south', 'north').replace('north-flipped', 'south')

    assert 'flipped' not in text

    return cell, text
    
class Kitti360CellDatasetMulti(Dataset):
    def __init__(self, base_path, scene_names, transform, split=None, shuffle_hints=False, flip_cells=False):
        self.scene_names = scene_names
        self.transform = transform
        self.split = split
        self.flip_cells = flip_cells
        self.datasets = [Kitti360CellDataset(base_path, scene_name, transform, split, shuffle_hints, flip_cells) for scene_name in scene_names]
        self.cells = [cell for dataset in self.datasets for cell in dataset.cells] # Gathering cells for retrieval plotting

        print(str(self))

    def __getitem__(self, idx):
        count = 0
        for dataset in self.datasets:
            idx_in_dataset = idx - count
            if idx_in_dataset < len(dataset):
                return dataset[idx_in_dataset]
            else:
                count += len(dataset)
        assert False

    def __len__(self):
        return np.sum([len(ds) for ds in self.datasets])

    def __repr__(self):
        return f'Kitti360CellDatasetMulti: {len(self.scene_names)} scenes, {len(self)} cells, split {self.split}, flip {self.flip_cells}'

    def get_known_words(self):
        known_words = []
        for ds in self.datasets:
            known_words.extend(ds.get_known_words())
        return list(np.unique(known_words))

    def get_known_classes(self):
        known_classes = []
        for ds in self.datasets:
            known_classes.extend(ds.get_known_classes())
        return list(np.unique(known_classes))

if __name__ == '__main__':
    base_path = './data/kitti360'
    folder_name = '2013_05_28_drive_0000_sync'    

    transform = T.FixedPoints(10000, replace=False, allow_duplicates=False)

    dataset = Kitti360CellDatasetMulti(base_path, [folder_name, ], transform)
    data = dataset[0]
