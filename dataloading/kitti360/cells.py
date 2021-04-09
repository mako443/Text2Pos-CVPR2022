from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

from datapreparation.kitti360.utils import CLASS_TO_LABEL, LABEL_TO_CLASS, CLASS_TO_MINPOINTS
from datapreparation.kitti360.imports import Object3d, Cell
from datapreparation.kitti360.drawing import show_pptk, show_objects, plot_cell
from dataloading.kitti360.base import Kitti360BaseDataset

class Kitti360CellDataset(Kitti360BaseDataset):
    def __getitem__(self, idx):
        cell = self.cells[idx]
        hints = self.hint_descriptions[idx]
        text = ' '.join(hints)
        return {
            'cells': cell,
            'texts': text,
            'cell_indices': idx,
            'scene_names': self.scene_name
        } 

    def __len__(self):
        return len(self.cells)

if __name__ == '__main__':
    base_path = './data/kitti360'
    folder_name = '2013_05_28_drive_0000_sync'    
    
    dataset = Kitti360CellDataset(base_path, folder_name)          
    data = dataset[0]

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=Kitti360BaseDataset.collate_fn)
    batch = next(iter(dataloader))