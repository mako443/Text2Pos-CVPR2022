from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2
from easydict import EasyDict

import torch
from torch.utils.data import Dataset, DataLoader

from datapreparation.kitti360.utils import CLASS_TO_LABEL, LABEL_TO_CLASS, CLASS_TO_MINPOINTS
from datapreparation.kitti360.imports import Object3d, Cell
from datapreparation.kitti360.drawing import show_pptk, show_objects, plot_cell
from dataloading.kitti360.base import Kitti360BaseDataset

class Kitti360PoseReferenceDataset(Kitti360BaseDataset):
    def __init__(self, base_path, scene_name, args, split=None):
        super().__init__(base_path, scene_name, split)
        self.pad_size = args.pad_size

    def __getitem__(self, idx):
        cell = self.cells[idx]
        hints = self.hint_descriptions[idx]
        
        descriptions = cell.descriptions
        cell_objects_dict = {obj.id: obj for obj in cell.objects}
        mentioned_ids = [descr.object_id for descr in descriptions]

        # Gather mentioned objects, matches and offsets
        objects, matches, offsets = [], [], []
        for i_descr, descr in enumerate(descriptions):
            hint_obj = cell_objects_dict[descr.object_id]
            objects.append(hint_obj)
            matches.append((i_descr, i_descr))
            offsets.append(cell.pose - hint_obj.closest_point)
        offsets = np.array(offsets)[:, 0:2]

        # Gather distractors
        for obj in cell.objects:
            if obj.id not in mentioned_ids:
                objects.append(obj)

        # Pad or cut-off distractors (CARE: the latter would use ground-truth data!)
        if len(objects) > self.pad_size:
            objects = objects[0 : self.pad_size]

        while len(objects) < self.pad_size:
            objects.append(Object3d(np.zeros((1,3)), np.zeros((1,3)), 'pad', -1))

        # Build matches and all_matches
        # The mentioned objects are always put in first, however, our geometric models have no knowledge of these indices
        matches = [(i, i) for i in range(len(descriptions))]
        all_matches = matches.copy()
        for i in range(len(objects)):
            if objects[i].id not in mentioned_ids:
                all_matches.append((i, len(descriptions))) # Match all distractors or pads to hints-side-bin
        matches, all_matches = np.array(matches), np.array(all_matches)
        assert np.sum(all_matches[:, 1] == len(descriptions)) == len(objects) - len(descriptions)

        return {
            'objects': objects,
            'hint_descriptions': hints,
            'matches': matches,
            'all_matches': all_matches,
            'poses': cell.pose,
            'offsets': np.array(offsets)
        }

    def __len__(self):
        return len(self.cells)
        
if __name__ == '__main__':
    base_path = './data/kitti360'
    folder_name = '2013_05_28_drive_0000_sync'    
    
    args = EasyDict(pad_size=8)

    dataset = Kitti360PoseReferenceDataset(base_path, folder_name, args)
    data = dataset[0]

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=Kitti360BaseDataset.collate_fn)
    batch = next(iter(dataloader))        