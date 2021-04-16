from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

from datapreparation.kitti360.utils import CLASS_TO_LABEL, LABEL_TO_CLASS, CLASS_TO_MINPOINTS, SCENE_NAMES
from datapreparation.kitti360.imports import Object3d, Cell
from datapreparation.kitti360.drawing import show_pptk, show_objects, plot_cell
from dataloading.kitti360.base import Kitti360BaseDataset

class Kitti360CellDataset(Kitti360BaseDataset):
    def __getitem__(self, idx):
        cell = self.cells[idx]
        hints = self.hint_descriptions[idx]
        text = ' '.join(hints)
        return {
            # 'cells': cell,
            'objects': cell.objects,
            'texts': text,
            # 'cell_indices': idx,
            'scene_names': self.scene_name
        } 

    def __len__(self):
        return len(self.cells)

class Kitti360CellDatasetMulti(Dataset):
    def __init__(self, base_path, scene_names, split=None):
        self.scene_names = scene_names
        self.split = split
        self.datasets = [Kitti360CellDataset(base_path, scene_name, split) for scene_name in scene_names]

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
        return f'Kitti360CellDatasetMulti: {len(self.scene_names)} scenes, {len(self)} cells, split {self.split}'

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
    # folder_name = '2013_05_28_drive_0000_sync'    
    dataset = Kitti360CellDatasetMulti(base_path, SCENE_NAMES)
    
    # dataset = Kitti360CellDataset(base_path, folder_name)          
    # data = dataset[0]

    # dataloader = DataLoader(dataset, batch_size=2, collate_fn=Kitti360BaseDataset.collate_fn)
    # batch = next(iter(dataloader))