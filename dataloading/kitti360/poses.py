from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2
from easydict import EasyDict

import torch
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T 

from datapreparation.kitti360.utils import CLASS_TO_LABEL, LABEL_TO_CLASS, CLASS_TO_MINPOINTS, CLASS_TO_INDEX, COLORS, COLOR_NAMES, SCENE_NAMES
from datapreparation.kitti360.imports import Object3d, Cell, Pose
from datapreparation.kitti360.drawing import show_pptk, show_objects, plot_cell
from dataloading.kitti360.base import Kitti360BaseDataset

def batch_object_points(objects: List[Object3d], transform):
    """Generates a PyG-Batch for the objects of a single cell.
    Note: Aggregating an entire batch of cells into a single PyG-Batch would exceed the limit of 256 sub-graphs.
    Note: The objects can be transformed / augmented freely, as their center-points are encoded separately.

    Args:
        objects (List[Object3d]): Cell objects
        transform: PyG-Transform
    """
    # CARE: Transforms not working with batches?! Doing it object-by-object here!
    data_list = [Data(x=torch.tensor(obj.rgb, dtype=torch.float), pos=torch.tensor(obj.xyz, dtype=torch.float)) for obj in objects]
    for i in range(len(data_list)):
        data_list[i] = transform(data_list[i])

    batch = Batch.from_data_list(data_list)
    return batch

def load_pose_and_cell(pose: Pose, cell: Cell, hints, pad_size, transform):
    descriptions = pose.descriptions
    cell_objects_dict = {obj.id: obj for obj in cell.objects}
    mentioned_ids = [descr.object_id for descr in descriptions]

    # Gather mentioned objects, matches and offsets
    objects, matches, offsets = [], [], []
    for i_descr, descr in enumerate(descriptions):
        hint_obj = cell_objects_dict[descr.object_id]
        objects.append(hint_obj)
        matches.append((i_descr, i_descr))
        # offsets.append(pose.pose - descr.object_closest_point)
        offsets.append(pose.pose - hint_obj.get_center())
    offsets = np.array(offsets)[:, 0:2]

    # Gather distractors
    for obj in cell.objects:
        if obj.id not in mentioned_ids:
            objects.append(obj)

    # Pad or cut-off distractors (CARE: the latter would use ground-truth data!)
    if len(objects) > pad_size:
        objects = objects[0 : pad_size]

    while len(objects) < pad_size:
        obj = Object3d.create_padding()
        objects.append(obj)

    # Build matches and all_matches
    # The mentioned objects are always put in first, however, our geometric models have no knowledge of these indices
    matches = [(i, i) for i in range(len(descriptions))]
    all_matches = matches.copy()
    for i in range(len(objects)):
        if objects[i].id not in mentioned_ids:
            all_matches.append((i, len(descriptions))) # Match all distractors or pads to hints-side-bin
    matches, all_matches = np.array(matches), np.array(all_matches)
    assert np.sum(all_matches[:, 1] == len(descriptions)) == len(objects) - len(descriptions)

    object_points = batch_object_points(objects, transform)

    object_class_indices = [CLASS_TO_INDEX[obj.label] for obj in objects]
    object_color_indices = [COLOR_NAMES.index(obj.get_color_text()) for obj in objects]        

    text = ' '.join(hints)

    return {
        'poses': pose,
        'cells': cell,
        'objects': objects,
        'object_points': object_points,
        'hint_descriptions': hints,
        'texts': text,
        'matches': matches,
        'all_matches': all_matches,
        'offsets': np.array(offsets),
        'object_class_indices': object_class_indices,
        'object_color_indices': object_color_indices
    }            

class Kitti360FineDataset(Kitti360BaseDataset):
    def __init__(self, base_path, scene_name, transform, args):
        super().__init__(base_path, scene_name)
        self.pad_size = args.pad_size
        self.transform = transform

    def __getitem__(self, idx):
        pose = self.poses[idx]
        cell = self.cells_dict[pose.cell_id]
        hints = self.hint_descriptions[idx]
        
        return load_pose_and_cell(pose, cell, hints, self.pad_size, self.transform)

    def __len__(self):
        return len(self.poses)

class Kitti360FineDatasetMulti(Dataset):
    def __init__(self, base_path, scene_names, transform, args):
        self.scene_names = scene_names
        self.datasets = [Kitti360FineDataset(base_path, scene_name, transform, args) for scene_name in scene_names]

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

    def __repr__(self):
        return f'Kitti360FineDatasetMulti: {len(self)} poses from {len(self.datasets)} scenes.'

    def __len__(self):
        return np.sum([len(ds) for ds in self.datasets])

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
    base_path = './data/k360_decouple'
    folder_name = '2013_05_28_drive_0003_sync'    
    
    args = EasyDict(pad_size=8, num_mentioned=6)    
    transform = T.Compose([T.FixedPoints(1024), T.NormalizeScale()])

    dataset = Kitti360FineDatasetMulti(base_path, [folder_name, ], transform, args)
    data = dataset[0]