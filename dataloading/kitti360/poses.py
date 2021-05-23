from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2
from easydict import EasyDict
from numpy.lib.function_base import flip

import torch
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T 

from datapreparation.kitti360.utils import CLASS_TO_LABEL, LABEL_TO_CLASS, CLASS_TO_MINPOINTS, CLASS_TO_INDEX, COLORS, COLOR_NAMES, SCENE_NAMES
from datapreparation.kitti360.imports import Object3d, Cell, Pose
from datapreparation.kitti360.drawing import show_pptk, show_objects, plot_cell
from dataloading.kitti360.base import Kitti360BaseDataset
from dataloading.kitti360.utils import batch_object_points, flip_pose_in_cell

def load_pose_and_cell(pose: Pose, cell: Cell, hints, pad_size, transform, flip_pose=False):
    assert pose.cell_id == cell.id

    descriptions = pose.descriptions
    cell_objects_dict = {obj.id: obj for obj in cell.objects}
    matched_ids = [descr.object_id for descr in descriptions if descr.is_matched]
    matched_objects = [cell_objects_dict[matched_id] for matched_id in matched_ids]

    assert len(pose.descriptions) - pose.get_number_unmatched() == len(matched_objects)

    # Hints and descriptions have to be in same order
    for descr, hint in zip(descriptions, hints):
        assert descr.object_label in hint

    # Gather offsets
    offsets = np.array([descr.offset_center for descr in descriptions])[:, 0:2]  

    # Gather matched objects
    objects, matches = [], [] # Matches as [(obj_idx, hint_idx)]
    for i_descr, descr in enumerate(descriptions):
        if descr.is_matched:
            hint_obj = cell_objects_dict[descr.object_id]
            assert hint_obj.instance_id == descr.object_instance_id
            objects.append(hint_obj)
            
            obj_idx = len(objects) - 1
            hint_idx = i_descr
            matches.append((obj_idx, hint_idx))
            
    # Gather distractors, i.e. remaining objects
    for obj in cell.objects:
        if obj.id not in matched_ids:
            objects.append(obj)
    if len(objects) != len(cell.objects):
        print([obj.id for obj in objects])
        print([obj.id for obj in cell.objects])
        print(matched_ids)
    assert len(objects) == len(cell.objects), f"Not all cell-objects have been gathered! {len(objects)}, {len(cell.objects)}, {cell.id}"

    # # Gather mentioned objects, matches and offsets
    # objects, matches, offsets = [], [], []
    # for i_descr, descr in enumerate(descriptions):
    #     hint_obj = cell_objects_dict[descr.object_id]
    #     objects.append(hint_obj)
    #     matches.append((i_descr, i_descr))
    #     # offsets.append(pose.pose - descr.object_closest_point)
    #     offsets.append(pose.pose - hint_obj.get_center())
    # offsets = np.array(offsets)[:, 0:2]

    # # Gather distractors
    # for obj in cell.objects:
    #     if obj.id not in mentioned_ids:
    #         objects.append(obj)

    # Pad or cut-off distractors (CARE: the latter would use ground-truth data!)
    if len(objects) > pad_size:
        objects = objects[0 : pad_size]

    while len(objects) < pad_size:
        obj = Object3d.create_padding()
        objects.append(obj)

    # Build matches and all_matches
    # The matched objects are always put in first, however, our geometric models have no knowledge of these indices as they are permutation invariant.
    all_matches = matches.copy()

    # Add unmatched hints
    for i_descr, descr in enumerate(descriptions):
        if not descr.is_matched:
            obj_idx = len(objects) # Match to objects-side bin
            hint_idx = i_descr
            all_matches.append((obj_idx, hint_idx))

    # Add unmatched objects
    for obj_idx, obj in enumerate(objects):
        if obj.id not in matched_ids:
            hint_idx = len(descriptions) # Match to hints-side bin
            all_matches.append((obj_idx, hint_idx))

    matches, all_matches = np.array(matches), np.array(all_matches)
    assert len(matches) == len(matched_ids)
    assert len(all_matches) == len(objects) + len(descriptions) - len(matches)
    assert np.sum(all_matches[:, 1] == len(descriptions)) == len(objects) - len(matched_ids) # Binned objects
    assert np.sum(all_matches[:, 0] == len(objects)) == len(descriptions) - len(matched_ids) # Binned hints

    # for i in range(len(objects)):
    #     if objects[i].id not in mentioned_ids:
    #         all_matches.append((i, len(descriptions))) # Match all distractors or pads to hints-side-bin
    # matches, all_matches = np.array(matches), np.array(all_matches)
    # assert np.sum(all_matches[:, 1] == len(descriptions)) == len(objects) - len(descriptions)

    text = ' '.join(hints)

    # TODO: flip here. This does not affect the the order and matches.
    # Flip objects, hint_descriptions, object_points, offsets, pose
    if flip_pose:
        if np.random.choice((True,False)):
            pose, cell, text, hints, offsets = flip_pose_in_cell(pose, cell, text, 1, hints, offsets) # Horizontal
        if np.random.choice((True,False)):
            pose, cell, text, hints, offsets = flip_pose_in_cell(pose, cell, text, -1, hints, offsets) # Vertical

    object_points = batch_object_points(objects, transform)

    object_class_indices = [CLASS_TO_INDEX[obj.label] for obj in objects]
    object_color_indices = [COLOR_NAMES.index(obj.get_color_text()) for obj in objects]        

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
    def __init__(self, base_path, scene_name, transform, args, flip_pose=False):
        super().__init__(base_path, scene_name)
        self.pad_size = args.pad_size
        self.transform = transform
        self.flip_pose = flip_pose

    def __getitem__(self, idx):
        pose = self.poses[idx]
        cell = self.cells_dict[pose.cell_id]
        hints = self.hint_descriptions[idx]
        
        return load_pose_and_cell(pose, cell, hints, self.pad_size, self.transform, self.flip_pose)

    def __len__(self):
        return len(self.poses)

    def collate_fn(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch         

class Kitti360FineDatasetMulti(Dataset):
    def __init__(self, base_path, scene_names, transform, args, flip_pose=False):
        self.scene_names = scene_names
        self.flip_pose = flip_pose
        self.datasets = [Kitti360FineDataset(base_path, scene_name, transform, args, flip_pose) for scene_name in scene_names]
        
        self.all_poses = [pose for dataset in self.datasets for pose in dataset.poses] # For stats
        self.all_cells = [cell for dataset in self.datasets for cell in dataset.cells] # For eval stats

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
        poses = np.array([pose.pose_w for pose in self.all_poses])
        num_poses = len(np.unique(poses, axis=0)) # CARE: Might be possible that is is slightly inaccurate if there are actually overlaps        
        return f'Kitti360FineDatasetMulti: {len(self)} descriptions for {num_poses} unique poses from {len(self.datasets)} scenes, {len(self.all_cells)} cells, flip: {self.flip_pose}.'

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
    base_path = './data/k360_cs30_cd30_pd30_shTrue'
    folder_name = '2013_05_28_drive_0003_sync'    
    
    args = EasyDict(pad_size=8, num_mentioned=6)    
    transform = T.Compose([T.FixedPoints(1024), T.NormalizeScale()])

    dataset = Kitti360FineDatasetMulti(base_path, [folder_name, ], transform, args)
    data = dataset[0]