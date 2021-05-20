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
from datapreparation.kitti360.imports import Object3d, Cell, Pose
from datapreparation.kitti360.drawing import show_pptk, show_objects, plot_cell, plot_pose_in_best_cell
from dataloading.kitti360.base import Kitti360BaseDataset
from dataloading.kitti360.utils import batch_object_points, flip_pose_in_cell

'''
Augmentations:
- hints order (care not to influence matches)
- pads to random objects and vice-versa
- flip cell
'''
class Kitti360CoarseDataset(Kitti360BaseDataset):
    def __init__(self, base_path, scene_name, transform, shuffle_hints=False, flip_poses=False):
        super().__init__(base_path, scene_name)
        self.shuffle_hints = shuffle_hints
        self.transform = transform
        self.flip_poses = flip_poses

    def __getitem__(self, idx):
        pose = self.poses[idx]
        cell = self.cells_dict[pose.cell_id]
        hints = self.hint_descriptions[idx]
        
        if self.shuffle_hints:
            hints = np.random.choice(hints, size=len(hints), replace=False)

        text = ' '.join(hints)

        # CARE: hints are currently not flipped! (Only the text.)
        if self.flip_poses:
            if np.random.choice((True, False)): # Horizontal
                pose, cell, text = flip_pose_in_cell(pose, cell, text, 1)
            if np.random.choice((True, False)): # Vertical
                pose, cell, text = flip_pose_in_cell(pose, cell, text, -1)                

        object_points = batch_object_points(cell.objects, self.transform)

        object_class_indices = [CLASS_TO_INDEX[obj.label] for obj in cell.objects]
        object_color_indices = [COLOR_NAMES.index(obj.get_color_text()) for obj in cell.objects]           

        return {
            'poses': pose,
            'cells': cell,
            'objects': cell.objects,
            'object_points': object_points,
            'texts': text,
            'cell_ids': pose.cell_id,
            'scene_names': self.scene_name,
            'object_class_indices': object_class_indices,
            'object_color_indices': object_color_indices            
        } 

    def __len__(self):
        return len(self.poses)
    
class Kitti360CoarseDatasetMulti(Dataset):
    def __init__(self, base_path, scene_names, transform, shuffle_hints=False, flip_poses=False):
        self.scene_names = scene_names
        self.transform = transform
        self.flip_poses = flip_poses
        self.datasets = [Kitti360CoarseDataset(base_path, scene_name, transform, shuffle_hints, flip_poses) for scene_name in scene_names]
        
        self.all_cells = [cell for dataset in self.datasets for cell in dataset.cells] # For cell-only dataset
        self.all_poses = [pose for dataset in self.datasets for pose in dataset.poses] # For eval


        cell_ids = [cell.id for cell in self.all_cells]
        assert len(np.unique(cell_ids)) == len(self.all_cells) # IDs should not repeat

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
        return f'Kitti360CellDatasetMulti: {len(self.scene_names)} scenes, {len(self)} poses, {len(self.all_cells)} cells, flip {self.flip_poses}'

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

    def get_cell_dataset(self):
        return Kitti360CoarseCellOnlyDataset(self.all_cells, self.transform)

class Kitti360CoarseCellOnlyDataset(Dataset):
    """Dataset to return only the cells for encoding during evaluation
    NOTE: The way the cells are read from the Cells-Only-Dataset, they may have been augmented differently during the actual training. Cells-Only does not flip and shuffle!
    TODO: This ok?
    """

    def __init__(self, cells: List[Cell], transform):
        super().__init__()

        self.cells = cells
        self.transform = transform

    def __getitem__(self, idx):
        cell = self.cells[idx]
        object_points = batch_object_points(cell.objects, self.transform)

        return {
            'cells': cell,
            'cell_ids': cell.id,
            'objects': cell.objects,
            'object_points': object_points
        }

    def __len__(self):
        return len(self.cells)        

if __name__ == '__main__':
    base_path = './data/k360_cs30_cd15_scY_pd10_pc1_spY_closest'
    folder_name = '2013_05_28_drive_0003_sync'    

    transform = T.FixedPoints(256)

    dataset = Kitti360CoarseDatasetMulti(base_path, [folder_name, ], transform, shuffle_hints=False, flip_poses=False)
    data = dataset[0]
    pose, cell, text = data['poses'], data['cells'], data['texts']
    offsets = np.array([descr.offset_closest for descr in pose.descriptions])
    hints = text.split('.')
    pose_f, cell_f, text_f, hints_f, offsets_f = flip_pose_in_cell(pose, cell, text, 1, hints=hints, offsets=offsets)