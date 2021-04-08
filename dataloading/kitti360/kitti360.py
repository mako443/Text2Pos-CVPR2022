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

class Kitti360BaseDataset(Dataset):
    def __init__(self, base_path, scene_name):
        self.scene_name = scene_name
        self.objects = pickle.load(open(osp.join(base_path, 'objects', f'{scene_name}.pkl'), 'rb'))
        self.cells = pickle.load(open(osp.join(base_path, 'cells', f'{scene_name}.pkl'), 'rb'))

        self.hint_descriptions = self.create_hint_descriptions(self.cells) # Gather here for get_known_words()

    def __getitem__(self, idx):
        raise Exception('Not implemented: abstract class.')

    def create_hint_descriptions(self, cells: List[Cell]):
        hint_descriptions = []
        for cell in cells:
            hints = []
            cell_objects_dict = {obj.id: obj for obj in cell.objects}
            for descr in cell.descriptions:
                obj = cell_objects_dict[descr.object_id]
                hints.append(f'The pose is {descr.direction} of a {obj.label}.')
            hint_descriptions.append(hints)

        return hint_descriptions

    def get_known_classes(self):
        classes = [obj.label for obj in self.scene_objects]
        classes.append('pad')
        return list(np.unique(classes))

    def get_known_words(self):
        words = []
        for hints in self.hint_descriptions:
            for hint in hints:
                words.extend(hint.replace('.','').replace(',','').lower().split())
        return list(np.unique(words))        

    def __len__(self):
        return len(self.cells)

    def collate_fn(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch

class Kitti360CellDataset(Kitti360BaseDataset):
    def __getitem__(self, idx):
        cell = self.cells[idx]
        hints = self.hint_descriptions[idx]
        text = ' '.join(hints)
        return {
            'objects': cell.objects,
            'descriptions': text,
            'scene_names': self.scene_name
        }      

if __name__ == '__main__':
    base_path = './data/kitti360'
    folder_name = '2013_05_28_drive_0000_sync'    
    
    dataset = Kitti360CellDataset(base_path, folder_name)          
    data = dataset[0]

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=Kitti360BaseDataset.collate_fn)
    batch = next(iter(dataloader))