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
    def __init__(self, base_path, scene_name, split=None):
        self.scene_name = scene_name
        self.objects = pickle.load(open(osp.join(base_path, 'objects', f'{scene_name}.pkl'), 'rb'))
        self.cells = pickle.load(open(osp.join(base_path, 'cells', f'{scene_name}.pkl'), 'rb'))

        print('CARE: removing small objects')
        self.objects = [o for o in self.objects if len(o.xyz)>512]

        if split is not None: # CARE: selects cells and objects, which aren't necessarily related!
            assert split in ('train', 'test')
            test_indices = (np.arange(np.max((len(self.objects), len(self.cells)))) % 5) == 0
            indices = test_indices if split=='test' else np.bitwise_not(test_indices)
            self.objects = [o for (i,o) in enumerate(self.objects) if indices[i]]      
            self.cells = [c for (i,c) in enumerate(self.cells) if indices[i]]      

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
        classes = [obj.label for obj in self.objects]
        classes.append('pad')
        return list(np.unique(classes))

    def get_known_words(self):
        words = []
        for hints in self.hint_descriptions:
            for hint in hints:
                words.extend(hint.replace('.','').replace(',','').lower().split())
        return list(np.unique(words))        

    def __len__(self):
        raise Exception('Not implemented: abstract class.')

    def collate_fn(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch


if __name__ == '__main__':
    base_path = './data/kitti360'
    folder_name = '2013_05_28_drive_0000_sync'    
    
    dataset = Kitti360BaseDataset(base_path, folder_name)          