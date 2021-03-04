import os
import os.path as osp
import sys
import time
import h5py
import json
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

import random

from datapreparation.imports import Object3D, DescriptionObject

class Semantic3dObjectReferanceDataset(Dataset):
    def __init__(self, path_numpy, path_scenes, num_distractors='all', split=None):
        self.path_numpy = path_numpy
        self.path_scenes = path_scenes
        self.num_distractors = num_distractors
        self.split = split
    
        self.scene_name = 'sg27_station5_intensity_rgb'

        #Load objects
        self.scene_objects     = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'objects.pkl'), 'rb'))
        self.list_descriptions = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'list_object_descriptions.pkl'), 'rb'))
        self.text_descriptions = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'text_object_descriptions.pkl'), 'rb'))
        self.hint_descriptions = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'hint_object_descriptions.pkl'), 'rb'))
        assert len(self.list_descriptions)==len(self.text_descriptions)==len(self.hint_descriptions)

        #Apply split
        if split is not None:
            assert split in ('train', 'test')
            test_indices = (np.arange(len(self.list_descriptions)) % 5) == 0
            indices = test_indices if split=='test' else np.bitwise_not(test_indices)
            self.list_descriptions = [ld for (i, ld) in enumerate(self.list_descriptions) if indices[i]]
            self.text_descriptions = [td for (i, td) in enumerate(self.text_descriptions) if indices[i]]
            self.hint_descriptions = [hd for (i, hd) in enumerate(self.hint_descriptions) if indices[i]]

        print(self)

    def __getitem__(self, idx):
        text_descr = self.text_descriptions[idx]
        list_descr = self.list_descriptions[idx]
        hint_descr = self.hint_descriptions[idx] #Hints are in the same order as list_descr
        assert len(list_descr) == len(hint_descr)

        mentioned_ids = [obj.id for obj in list_descr] #Including the target
        target_idx, target_id, target_class = [(i, obj.id, obj.label) for (i, obj) in enumerate(list_descr) if obj.is_target][0] 

        mentioned_objects = [obj for obj in self.scene_objects if obj.id in mentioned_ids] #Including target

        #Gather the distractor objects
        if self.num_distractors == 'all':
            distractor_objects = [obj for obj in self.scene_objects if obj.id not in mentioned_ids]
            random.shuffle(distractor_objects)
        elif self.num_distractors > 0:
            distractor_objects = np.random.choice([obj for obj in self.scene_objects if obj.id not in mentioned_ids], size=self.num_distractors, replace=False)
        else:
            distractor_objects = []

        #Gather the classes, positions and colors (colors ∈ [0, 1])
        mentioned_objects_classes = [obj.label for obj in mentioned_objects]
        mentioned_objects_positions = np.array([obj.center[0:2] for obj in mentioned_objects])
        mentioned_objects_colors = np.array([obj.color for obj in mentioned_objects])
        if len(distractor_objects) > 0:
            distractor_objects_classes = [obj.label for obj in distractor_objects]
            distractor_objects_positions = np.array([obj.center[0:2] for obj in distractor_objects])
            distractor_objects_colors = np.array([obj.color for obj in distractor_objects])
        else:
            distractor_objects_classes = []
            distractor_objects_positions = np.array([]).reshape((0,2))
            distractor_objects_colors = np.array([]).reshape((0,3))

        #Gather the offset vectors
        for obj in self.scene_objects:
            if obj.id == target_id:
                target_center = obj.center[0:2]
        offset_vectors = target_center - np.vstack((mentioned_objects_positions, distractor_objects_positions))

        mentioned_objects_mask = np.zeros(len(mentioned_objects_classes) + len(distractor_objects_classes), dtype=np.int64)
        mentioned_objects_mask[0:len(mentioned_objects_classes)] = 1

        #Build the matches
        matches = []
        all_matches = [] #Matches as [(obj_idx, hint_idx), (...)]
        for i in range(len(mentioned_objects)):
            all_matches.append((i, i))
            matches.append((i, i))
        bin_idx = len(mentioned_objects)
        for i in range(len(distractor_objects)):
            all_matches.append((bin_idx + i, bin_idx))
        matches = np.array(matches)
        all_matches = np.array(all_matches)
        assert len(matches) == len(mentioned_objects)
        assert len(all_matches) == len(mentioned_objects) + len(distractor_objects) and np.sum(all_matches[:, 1]==bin_idx) == len(distractor_objects)

        #Always stacks mentioned then distractors
        return {
            'target_idx': target_idx,
            'objects_classes': mentioned_objects_classes + distractor_objects_classes,
            'objects_positions': np.vstack((mentioned_objects_positions, distractor_objects_positions)),
            'objects_colors': np.vstack((mentioned_objects_colors, distractor_objects_colors)),
            'text_descriptions': text_descr,
            'hint_descriptions': hint_descr,
            'target_classes': target_class,
            'num_mentioned': len(list_descr),
            'num_distractors': len(distractor_objects_classes),
            # 'offset_vectors': offset_vectors,
            'mentioned_objects_mask': mentioned_objects_mask,
            'matches': matches,
            'all_matches': all_matches #Collate as list of np.ndarray because the number of matches might vary.
        }

    # Gather lists in outer list, stack arrays along new dim. Batch comes as list of dicts.
    def collate_fn(batch):
        data = {}
        for k, v in batch[0].items():
            if k in ('matches', 'all_matches'):
                data[k] = [batch[i][k] for i in range(len(batch))]
            elif isinstance(v, list) or isinstance(v, str):
                data[k] = [batch[i][k] for i in range(len(batch))]
            elif isinstance(v, np.ndarray):
                data[k] = np.stack([batch[i][k] for i in range(len(batch))])
            elif isinstance(v, int):
                data[k] = [batch[i][k] for i in range(len(batch))]
            else:
                raise Exception(f'Unexpected type {str(type(v))} for key {k}')
        return data


    def __len__(self):
        return len(self.list_descriptions)

    def __repr__(self):
        return f'Semantic3dObjectReferanceDataset: {len(self.scene_objects)} objects and {len(self.text_descriptions)} text descriptions, <{self.num_distractors}> distractors, split {self.split}.'

    def get_known_classes(self):
        classes = [obj.label for obj in self.scene_objects]
        return list(np.unique(classes))
    
    def get_known_words(self):
        words = []
        for d in self.text_descriptions:
            words.extend(d.replace('.','').replace(',','').lower().split())
        for hints in self.hint_descriptions:
            for hint in hints:
                words.extend(hint.replace('.','').replace(',','').lower().split())
        return list(np.unique(words))
        

if __name__ == "__main__":
    dataset = Semantic3dObjectReferanceDataset('./data/numpy_merged/', './data/semantic3d', num_distractors=2)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=Semantic3dObjectReferanceDataset.collate_fn)
    data = dataset[0]
    batch = next(iter(dataloader))

