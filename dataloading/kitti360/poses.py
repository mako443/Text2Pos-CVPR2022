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
from datapreparation.kitti360.imports import Object3d, Cell
from datapreparation.kitti360.drawing import show_pptk, show_objects, plot_cell
from dataloading.kitti360.base import Kitti360BaseDataset

# TODO: remove
class Kitti360PoseReferenceMockDataset(Dataset):
    def __init__(self, args, length=1024, fixed_seed=False):
        self.pad_size = args.pad_size
        self.num_mentioned = args.num_mentioned
        self.length = length
        self.fixed_seed = fixed_seed

        self.classes = [c for c in CLASS_TO_INDEX if c!='pad']
        self.colors = COLORS
        self.color_names = COLOR_NAMES

        # self.reset_seed()

    # def reset_seed(self):
    #     """Resets the np random seed for re-producible results.
    #     Done initially and can be done before every epoch for consistent data.
    #     """
    #     np.random.seed(4096)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.fixed_seed:
            np.random.seed(idx)

        # Create random objects in the cell
        # num_distractors = self.num_distractors #np.random.randint(self.pad_size - self.num_mentioned) if self.pad_size > self.num_mentioned else 0
        num_distractors = np.random.randint(self.pad_size - self.num_mentioned) if self.pad_size > self.num_mentioned else 0
        objects = []
        for i in range(self.pad_size):
            if i >= self.num_mentioned + num_distractors:
                xyz = np.zeros((1, 3))
                label = 'pad'
                rgb = np.zeros((1, 3))
            else:
                xyz = np.random.rand(1, 3) # Center is inferred from this
                rgb = np.random.rand(1, 3)
                label = np.random.choice(self.classes)
            objects.append(Object3d(xyz, rgb, label, None))

        # create the pose somewhere in the cell
        pose = np.random.rand(3)

        # Random shift an object close to the pose for on-top
        if np.random.choice((True, False)):
            idx = np.random.randint(self.num_mentioned + num_distractors)
            objects[idx].xyz = np.array(pose + np.random.randn(3)*0.01).reshape((1,3))

        # Give hints for the <num_mentioned> closest objects
        hints = []
        matches = [] # (i,j) entry means obj-i matches hint-j
        distances = np.linalg.norm(pose[0:2] - np.array([obj.get_closest_point(pose) for obj in objects])[:, 0:2], axis=1) # Distance only x&y

        distances[self.num_mentioned + num_distractors : ] = np.inf # Remove the padding objects
        sorted_indices = np.argsort(distances)

        offset_vectors = []

        for hint_idx, obj_idx in enumerate(sorted_indices[0 : self.num_mentioned]):
            obj = objects[obj_idx]
            # color_dists = np.linalg.norm(obj.color - COLORS, axis=1)
            # color_text = COLOR_NAMES[np.argmin(color_dists)]
            color_dists = np.linalg.norm(obj.get_color_rgb() - self.colors, axis=1)
            color_text = self.color_names[np.argmin(color_dists)]            

            obj2pose = pose - obj.closest_point
            offset_vectors.append(obj2pose[0:2])
            if np.linalg.norm(obj2pose[0:2]) < 0.015: # Say 'on-top' if the object is very close (e.g. road), only calculated in x-y-plane!
                direction = 'on-top'
            else:
                if abs(obj2pose[0])>=abs(obj2pose[1]) and obj2pose[0]>=0: direction='east'
                if abs(obj2pose[0])>=abs(obj2pose[1]) and obj2pose[0]<=0: direction='west'
                if abs(obj2pose[0])<=abs(obj2pose[1]) and obj2pose[1]>=0: direction='north'
                if abs(obj2pose[0])<=abs(obj2pose[1]) and obj2pose[1]<=0: direction='south' 

            hints.append(f'The pose is {direction} of a {color_text} {obj.label}')
            matches.append((obj_idx, hint_idx))

        # offset_vectors = np.array(offset_vectors) / np.linalg.norm(offset_vectors, axis=1).reshape((-1,1)) # Norm to have offset directions (for now)

        # Create <matches> and <all_matches>
        all_matches = matches.copy()
        matches = np.array(matches)
        for obj_idx in range(len(objects)):
            if obj_idx not in matches[:, 0]: # If the object is not mentioned, i.e. in matches
                all_matches.append((obj_idx, self.num_mentioned)) # Then match it to the hints-side bin
        all_matches = np.array(all_matches)
        assert len(matches) == self.num_mentioned and np.sum(all_matches[:, 1] == self.num_mentioned) == self.pad_size - self.num_mentioned
        # assert len(all_matches) == self.num_mentioned + self.num_distractors and np.sum(all_matches[:, 1]==self.num_mentioned) == self.num_distractors

        return {
            'objects': objects,
            'hint_descriptions': hints,
            'texts': ' '.join(hints),
            'num_mentioned': self.num_mentioned,
            'num_distractors': num_distractors,
            'matches': matches,
            'all_matches': all_matches,
            'poses': pose,
            'offsets': np.array(offset_vectors)
        }

    def collate_fn(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch                 

    def get_known_classes(self):
        return self.classes + ['pad', ]   

    def get_known_words(self):
        known_words = []
        for i in range(50):
            data = self[i]
            for hint in data['hint_descriptions']:
                known_words.extend(hint.replace('.','').replace(',','').lower().split())
        return list(np.unique(known_words))  

def batch_object_points(objects: List[Object3d], transform):
    """Generates a PyG-Batch for the objects of a single cell.
    Note: Aggregating an entire cell-batch into a single PyG-Batch would exceed the limit of 256 sub-graphs.
    Note: The objects can be transformed / augmented freely, as their center-points are encoded separately.

    Args:
        objects (List[Object3d]): Cell objects
        transform ([type]): PyG-Transform
    """
    # CARE: Transforms not working with batches?! Doing it object-by-object here!
    data_list = [Data(x=torch.tensor(obj.rgb, dtype=torch.float), pos=torch.tensor(obj.xyz, dtype=torch.float)) for obj in objects]
    for i in range(len(data_list)):
        data_list[i] = transform(data_list[i])

    batch = Batch.from_data_list(data_list)
    return batch


def load_cell_data(cell, hints, pad_size, transform):
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
        if len(objects) > pad_size:
            objects = objects[0 : pad_size]

        while len(objects) < pad_size:
            # obj = Object3d(np.zeros((1,3)), np.zeros((1,3)), 'pad', -1)
            # _ = obj.get_closest_point(cell.pose) # run to set the point
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

        return {
            'objects': objects,
            'object_points': object_points,
            'hint_descriptions': hints,
            'matches': matches,
            'all_matches': all_matches,
            'poses': cell.pose,
            'offsets': np.array(offsets),
            'cells': cell
        }            

class Kitti360PoseReferenceDataset(Kitti360BaseDataset):
    def __init__(self, base_path, scene_name, transform, args, split=None):
        super().__init__(base_path, scene_name, split)
        self.pad_size = args.pad_size
        self.transform = transform

    def __getitem__(self, idx):
        cell = self.cells[idx]
        hints = self.hint_descriptions[idx]
        
        return load_cell_data(cell, hints, self.pad_size, self.transform)

    def __len__(self):
        return len(self.cells)

class Kitti360PoseReferenceDatasetMulti(Dataset):
    def __init__(self, base_path, scene_names, transform, args, split=None):
        self.scene_names = scene_names
        self.split = split
        self.datasets = [Kitti360PoseReferenceDataset(base_path, scene_name, transform, args, split) for scene_name in scene_names]

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
        return f'Kitti360PoseReferenceDatasetMulti: {len(self)} poses/cells from {len(self.scene_names)} scenes, split: {self.split}'

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
    base_path = './data/kitti360'
    folder_name = '2013_05_28_drive_0000_sync'    
    
    args = EasyDict(pad_size=8, num_mentioned=6)    
    transform = T.Compose([T.FixedPoints(1024), T.NormalizeScale()])
    dataset = Kitti360PoseReferenceDataset(base_path, folder_name, transform, args)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=Kitti360PoseReferenceDataset.collate_fn)
    data = dataset[0]
    batch = next(iter(dataloader))