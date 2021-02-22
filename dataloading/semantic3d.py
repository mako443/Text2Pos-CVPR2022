import os
import os.path as osp
import sys
import time
import h5py
import json
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

from datapreparation.imports import Object3D, DescriptionObject

class Semantic3dObjectReferanceDataset(Dataset):
    def __init__(self, path_numpy, path_scenes, num_distractors='all'):
        self.path_numpy = path_numpy
        self.path_scenes = path_scenes
        self.num_distractors = num_distractors
    
        self.scene_name = 'sg27_station5_intensity_rgb'

        #Load objects
        self.scene_objects     = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'objects.pkl'), 'rb'))
        self.list_descriptions = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'list_object_descriptions.pkl'), 'rb'))
        self.text_descriptions = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'text_object_descriptions.pkl'), 'rb'))
        assert len(self.list_descriptions)==len(self.text_descriptions)

        print(self)

    def __getitem__(self, idx):
        text_descr = self.text_descriptions[idx]
        list_descr = self.list_descriptions[idx]

        mentioned_ids = [obj.id for obj in list_descr] #Including the target
        target_idx, target_id, target_class = [(i, obj.id, obj.label) for (i, obj) in enumerate(list_descr) if obj.is_target][0] 

        mentioned_objects = [obj for obj in self.scene_objects if obj.id in mentioned_ids] #Including target

        
        #Gather the distractor objects
        if self.num_distractors == 'all':
            distractor_objects = [obj for obj in self.scene_objects if obj.id not in mentioned_ids]
        elif self.num_distractors > 0:
            distractor_objects = np.random.choice([obj for obj in self.scene_objects if obj.id not in mentioned_ids], size=self.num_distractors, replace=False)
        else:
            distractor_objects = []

        mentioned_objects_classes = [obj.label for obj in mentioned_objects]
        mentioned_objects_positions = np.array([obj.center[0:2] for obj in mentioned_objects])
        distractor_objects_classes = [obj.label for obj in distractor_objects]
        distractor_objects_positions = np.array([obj.center[0:2] for obj in distractor_objects])

        #Always stacks mentioned then distractors
        return {
            'target_idx': target_idx,
            'objects_classes': mentioned_objects_classes + distractor_objects_classes,
            'objects_positions': np.vstack((mentioned_objects_positions, distractor_objects_positions)),
            'text_descriptions': text_descr,
            'target_classes': target_class,
            'description_lengths': len(list_descr)
        }

        # return { 
        #     'target_idx': target_idx,
        #     'mentioned_objects_classes': mentioned_objects_classes,
        #     'mentioned_objects_positions': mentioned_objects_positions,
        #     'distractor_objects_classes': distractor_objects_classes,
        #     'distractor_objects_positions': distractor_objects_positions,
        #     'text_descriptions': text_descr,
        #     'target_classes': target_class
        # }

    # Gather lists in outer list, stack arrays along new dim. Batch comes as list of dicts.
    def collate_fn(batch):
        data = {}
        for k, v in batch[0].items():
            if isinstance(v, list) or isinstance(v, str):
                data[k] = [batch[i][k] for i in range(len(batch))]
            elif isinstance(v, np.ndarray):
                data[k] = np.stack([batch[i][k] for i in range(len(batch))])
            elif isinstance(v, int):
                data[k] = [batch[i][k] for i in range(len(batch))]
            else:
                raise Exception('Not implemented'+str(type(v)))
        return data


    def __len__(self):
        return len(self.list_descriptions)

    def __repr__(self):
        return f'Semantic3dObjectReferanceDataset: {len(self.scene_objects)} objects and {len(self.text_descriptions)} text descriptions.'

    def get_known_classes(self):
        classes = [obj.label for obj in self.scene_objects]
        return list(np.unique(classes))
    
    def get_known_words(self):
        words = []
        for d in self.text_descriptions:
            words.extend(d.replace('.','').replace(',','').lower().split())
        return list(np.unique(words))
        

if __name__ == "__main__":
    dataset = Semantic3dObjectReferanceDataset('./data/numpy_merged/', './data/semantic3d', num_distractors=1)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=Semantic3dObjectReferanceDataset.collate_fn)
    data = dataset[0]
    batch = next(iter(dataloader))

