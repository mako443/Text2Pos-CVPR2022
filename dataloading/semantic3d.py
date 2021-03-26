import os
import os.path as osp
import sys
import time
import h5py
import json
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

# import torch_geometric.transforms as T
# from torch_geometric.data import Data as PyG_Data, DataLoader as PyG_DataLoader

import random
import cv2

from datapreparation.imports import Object3D, DescriptionObject, COMBINED_SCENE_NAMES, COLORS, COLOR_NAMES
from datapreparation.drawing import draw_cells

'''
Mock data for explicit matching (object reference: one object is target, some other objects are mentioned, rest are distractors)
'''
class Semantic3dObjectReferanceMockDataset(Dataset):
    def __init__(self, num_mentioned, num_distractors, length=32):
        self.num_mentioned = num_mentioned
        self.num_distractors = num_distractors
        self.length = length

        self.classes = ['high vegetation', 'low vegetation', 'buildings', 'hard scape', 'cars']

    def __getitem__(self, idx):
        # Create random objects in the cell
        objects = []
        for i in range(self.num_mentioned + self.num_distractors):
            points_w = np.random.rand(1,3) # Center is inferred from this
            label = np.random.choice(self.classes)
            color = np.random.rand(3)
            objects.append(Object3D.from_mock_data({"points_w": points_w, "label": label, "color": color, "id": i}))

        # Create a hint description for each mentioned object
        hints = []
        for i in range(self.num_mentioned):
            obj = objects[i]
            target = objects[0]
            color_dists = np.linalg.norm(obj.color - COLORS, axis=1)
            color_text = COLOR_NAMES[np.argmin(color_dists)]

            if i == 0: # Target
                hints.append(f"The target is a {color_text} {obj.label}.")
            else: # Mentioned
                obj_to_target = target.center - obj.center
                if abs(obj_to_target[0])>=abs(obj_to_target[1]) and obj_to_target[0]>=0: direction='east'
                if abs(obj_to_target[0])>=abs(obj_to_target[1]) and obj_to_target[0]<=0: direction='west'
                if abs(obj_to_target[0])<=abs(obj_to_target[1]) and obj_to_target[1]>=0: direction='north'
                if abs(obj_to_target[0])<=abs(obj_to_target[1]) and obj_to_target[1]<=0: direction='south' 

                hints.append(f"It is {direction} of a {color_text} {obj.label}.")
        
        # Create <matches> and <all_matches>
        matches = [(i, i) for i in range(self.num_mentioned)]
        all_matches = [(i, i) for i in range(self.num_mentioned)]
        for i in range(self.num_distractors): # Match all distractor objects to the hints-side bin
            all_matches.append((self.num_mentioned + i, self.num_mentioned)) 
        matches, all_matches = np.array(matches), np.array(all_matches)
        assert len(matches) == self.num_mentioned
        assert len(all_matches) == self.num_mentioned + self.num_distractors and np.sum(all_matches[:, 1]==self.num_mentioned) == self.num_distractors

        return {
            'target_idx': 0,
            'objects': objects,
            'hint_descriptions': hints,
            'target_classes': target.label,
            'num_mentioned': self.num_mentioned,
            'num_distractors': self.num_distractors,
            'matches': matches,
            'all_matches': all_matches #Collate as list of np.ndarray because the number of matches might vary.
        }

    def __len__(self):
        return self.length        

    def collate_fn(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch        

    def get_known_classes(self):
        return self.classes

    def get_known_words(self):
        known_words = []
        for i in range(50):
            data = self[0]
            for hint in data['hint_descriptions']:
                known_words.extend(hint.replace('.','').replace(',','').lower().split())
        return list(np.unique(known_words))

'''
Mock data for explicit matching (pose reference: pose is described relative to some mentioned objects, rest are distractors)
'''
class Semantic3dPoseReferanceMockDataset(Semantic3dObjectReferanceMockDataset):
    def __init__(self, num_mentioned, pad_size, length=32):
        super().__init__(num_mentioned, -1, length)
        self.pad_size = pad_size

        self.classes = ['high vegetation', 'low vegetation', 'buildings', 'hard scape', 'cars'] # Added <pad> 

    def get_known_classes(self):
        return self.classes + ['pad', ]        

    def __getitem__(self, idx):
        # Create random objects in the cell
        num_distractors = np.random.randint(self.pad_size - self.num_mentioned) if self.pad_size > self.num_mentioned else 0
        objects = []
        for i in range(self.num_mentioned + num_distractors):
            points_w = np.random.rand(1,3) # Center is inferred from this
            label = np.random.choice(self.classes)
            color = np.random.rand(3)
            objects.append(Object3D.from_mock_data({"points_w": points_w, "label": label, "color": color, "id": i}))              

        # create the pose somewhere in the cell
        pose = np.random.rand(3)

        # Give hints for the <num_mentioned> closest objects
        hints = []
        matches = [] # (i,j) entry means obj-i matches hint-j
        distances = np.linalg.norm(pose[0:2] - np.array([obj.center for obj in objects])[:, 0:2], axis=1) # Distance only x&y
        sorted_indices = np.argsort(distances)
        for hint_idx, obj_idx in enumerate(sorted_indices[0 : self.num_mentioned]):
            obj = objects[obj_idx]
            color_dists = np.linalg.norm(obj.color - COLORS, axis=1)
            color_text = COLOR_NAMES[np.argmin(color_dists)]
            
            obj_to_pose = pose - obj.center
            if abs(obj_to_pose[0])>=abs(obj_to_pose[1]) and obj_to_pose[0]>=0: direction='east'
            if abs(obj_to_pose[0])>=abs(obj_to_pose[1]) and obj_to_pose[0]<=0: direction='west'
            if abs(obj_to_pose[0])<=abs(obj_to_pose[1]) and obj_to_pose[1]>=0: direction='north'
            if abs(obj_to_pose[0])<=abs(obj_to_pose[1]) and obj_to_pose[1]<=0: direction='south' 

            hints.append(f'The pose is {direction} of a {color_text} {obj.label}')
            matches.append((obj_idx, hint_idx))

        # Create pad objects
        while len(objects) < self.pad_size:
            points_w = np.zeros((1, 3))
            label = 'pad'
            color = np.zeros((1, 3))
            objects.append(Object3D.from_mock_data({"points_w": points_w, "label": label, "color": color, "id": -1}))              

        # Create <matches> and <all_matches>
        all_matches = matches.copy()
        matches = np.array(matches)
        for obj_idx in range(len(objects)):
            if obj_idx not in matches[:, 0]: # If the object is not mentioned, i.e. in matches
                all_matches.append((obj_idx, self.num_mentioned)) # Then match it to the hints-side bin
        all_matches = np.array(all_matches)
        assert len(matches) == self.num_mentioned
        # assert len(all_matches) == self.num_mentioned + .num_distractors and np.sum(all_matches[:, 1]==self.num_mentioned) == self.num_distractors

        return {
            'objects': objects,
            'hint_descriptions': hints,
            'num_mentioned': self.num_mentioned,
            'num_distractors': self.num_distractors,
            'matches': matches,
            'all_matches': all_matches,
            'poses': pose
        }

'''
Currently uses cell-oracle: cell is placed so that it exactly fits all mentioned objects, retains all distractors
'''
class Semantic3dPoseReferanceDataset(Dataset):
    def __init__(self, path_numpy, path_scenes, scene_name, pad_size, split=None):
        self.path_numpy = path_numpy
        self.path_scenes = path_scenes
        self.split = split
        self.pad_size = pad_size
    
        self.scene_name = scene_name

        #Load objects
        self.scene_objects = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'objects.pkl'), 'rb'))
        self.poses = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'poses.pkl'), 'rb'))
        self.pose_descriptions = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'pose_descriptions.pkl'), 'rb'))
        assert len(self.poses)==len(self.pose_descriptions)      

        if split is not None:
            assert split in ('train', 'test')
            test_indices = (np.arange(len(self.poses)) % 5) == 0
            indices = test_indices if split=='test' else np.bitwise_not(test_indices)
            self.poses = [p for (i, p) in enumerate(self.poses) if indices[i]]            
            self.pose_descriptions = [p for (i, p) in enumerate(self.pose_descriptions) if indices[i]]

        self.create_hint_descriptions() # Create here to have known words

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

    def create_hint_descriptions(self):
        self.hint_descriptions = []
        for pose_description in self.pose_descriptions:
            hints = []
            for do in pose_description:
                hints.append(f'The pose is {do.direction} of a {do.color_text} {do.label}.')
            self.hint_descriptions.append(hints)

    def __getitem__(self, idx):
        pose, pose_description, hint_descriptions = self.poses[idx], self.pose_descriptions[idx], self.hint_descriptions[idx]
        objects_dict = {obj.id: obj for obj in self.scene_objects}
        mentioned_ids = [do.id for do in pose_description]
        mentioned_objects = [objects_dict[id] for id in mentioned_ids]

        # Build cell on the fly - currently as oracle that perfectly fits all mentioned objects
        mentioned_bboxes = [obj.aligned_bbox for obj in mentioned_objects]
        mentioned_bboxes = np.array(mentioned_bboxes)
        mentioned_bboxes[:, 3:6] += mentioned_bboxes[:, 0:3] # now [x0, y0, z0, x1, y1, z1]
        cell_bbox = np.hstack((np.min(mentioned_bboxes[:, 0:3], axis=0), np.max(mentioned_bboxes[:, 3:6], axis=0)))
        # cell_mean = 0.5 * (cell_bbox[0:3] + cell_bbox[3:6])
        cell_sizes = cell_bbox[3:6] - cell_bbox[0:3]
        assert np.all(cell_sizes>0)

        cell_objects = []

        # Gather the mentioned objects
        # CARE: using "mock" Object3D here, not cell objects
        for obj in mentioned_objects:
            cell_object = Object3D.from_mock_data({'label': obj.label,  'id': obj.id, 'points_w': obj.points_w, 'color': obj.color})
            cell_objects.append(cell_object)

        # Gather the distractors (objects that are at least 1/3 in cell)
        for obj in self.scene_objects:
            if obj.id in mentioned_ids:
                continue # Only gathering distractors
            if len(cell_objects) == self.pad_size: # CARE: removing some distractors if over pad_size, this uses ground-truth data!
                break 

            mask = np.bitwise_and.reduce((
                obj.points_w[:, 0] >= cell_bbox[0],
                obj.points_w[:, 1] >= cell_bbox[1],
                obj.points_w[:, 2] >= cell_bbox[2],
                obj.points_w[:, 0] <= cell_bbox[3],
                obj.points_w[:, 1] <= cell_bbox[4],
                obj.points_w[:, 2] <= cell_bbox[5],
            ))
            points_in_cell = obj.points_w[mask]
            points_in_cell_color = obj.points_w_color[mask]

            if len(points_in_cell) / len(obj.points_w) >= 1/3:
                cell_object = Object3D.from_mock_data({'label': obj.label,  'id': obj.id, 'points_w': obj.points_w, 'color': obj.color})
                cell_objects.append(cell_object)

        # Norming all points to [0, 1] relative to cell
        # CARE: not all [0, 1] because distractors could be partially outside
        for obj in cell_objects:
            obj.points_w = (obj.points_w - cell_bbox[0:3]) / cell_sizes
        pose = (pose.eye - cell_bbox[0:3]) / cell_sizes

        # Padding objects (care: after norming)
        # Create pad objects
        while len(cell_objects) < self.pad_size:
            points_w = np.zeros((1, 3))
            label = 'pad'
            color = np.zeros((1, 3))
            cell_objects.append(Object3D.from_mock_data({"points_w": points_w, "label": label, "color": color, "id": -1}))            

        # Create <matches> and <all_matches>
        matches = [(i, i) for i in range(len(mentioned_objects))]
        all_matches = [(i, i) for i in range(len(mentioned_objects))]
        for i in range(len(cell_objects)):
            if i<len(matches):
                continue
            all_matches.append((i, len(mentioned_objects))) # Assign distractors to hints-side bin

        matches, all_matches = np.array(matches), np.array(all_matches)
        assert np.sum(all_matches[:, 1]==len(mentioned_objects)) == len(cell_objects)-len(mentioned_objects)

        return {
            'objects': cell_objects,
            'hint_descriptions': hint_descriptions,
            'matches': matches,
            'all_matches': all_matches,
            'poses': pose
        }

    def __len__(self):
        return len(self.poses)

    def collate_fn(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch

'''
DEPRECATED
Use free-form poses datasets instead
'''
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
        distractor_objects = list(distractor_objects)

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
            'objects': mentioned_objects + distractor_objects,
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

class Semantic3dCellRetrievalDataset(Dataset):
    def __init__(self, path_numpy, path_scenes, scene_name, mention_features, split=None):       
        self.path_numpy = path_numpy
        self.path_scenes = path_scenes
        self.mention_features = mention_features
    
        # self.scene_name = 'bildstein_station1_xyz_intensity_rgb'
        self.scene_name = scene_name

        #Load objects
        self.scene_objects = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'objects.pkl'), 'rb'))
        self.cells = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'cell_object_descriptions.pkl'), 'rb'))

        #Build descriptions (do it here to have known words available)
        self.cell_texts = [self.build_description(cell) for cell in self.cells]

        #Possibly apply split
        if split is not None:
            assert split in ('train', 'test')
            test_indices = (np.arange(len(self.cells)) % 5) == 0
            indices = test_indices if split=='test' else np.bitwise_not(test_indices)    
            self.cells = [c for (i, c) in enumerate(self.cells) if indices[i]]        
            self.cell_texts = [t for (i, t) in enumerate(self.cell_texts) if indices[i]]        

        self.mean_obj = np.mean([len(cell['objects']) for cell in self.cells])
        # print(f'Semantic3dCellRetrievalDataset ({self.scene_name}): {len(self)} cells with {mean_obj: 0.2f} objects avg, features: {mention_features}')

    def __getitem__(self, idx):
        cell = self.cells[idx]
        bbox = cell['bbox']
        cell_mean = 0.5*(bbox[0:2] + bbox[2:4])
        cell_size = np.mean((bbox[2] - bbox[0], bbox[3] - bbox[1]))

        cell_objects = cell['objects']
    
        description = self.cell_texts[idx]

        #Select a cell index for negative objects in Triplet-training
        negative_indices = [i for i in range(len(self)) if i!=idx and len(self.cell_texts)!=len(description)]
        negative_objects = self.cells[np.random.choice(negative_indices)]['objects']

        return {'objects': cell_objects,
                'negative_objects': negative_objects,
                'descriptions': description,
                'scene_names': self.scene_name }

    def build_description(self, cell):
        cell_objects = cell['objects']
        bbox = cell['bbox']
        cell_mean = 0.5*(bbox[0:2] + bbox[2:4])
        cell_size = np.mean((bbox[2] - bbox[0], bbox[3] - bbox[1]))

        directions = np.array([ [0, 1],
                                [0,-1],
                                [ 1, 0],
                                [-1, 0],
                                [ 0, 0],]) * cell_size/3
        direction_names = ['north', 'south', 'east', 'west', 'center']

        text = ""
        for i, obj in enumerate(cell_objects):
            if i==0:
                text += 'The cell has a '
            else:
                text += ' and a '

            if 'color' in self.mention_features:
                text += f'{obj.color_text} '

            if 'class' in self.mention_features:
                text += f'{obj.label} '

            if 'position' in self.mention_features:
                direction_diffs = np.linalg.norm(directions - obj.center_in_cell[0:2], axis=1)
                direction = direction_names[np.argmin(direction_diffs)]
                text += f'in the {direction}'
        text += '.'

        return text

    def __len__(self):
        return len(self.cells)    
    
    def get_known_classes(self):
        classes = []
        for cell in self.cells:
            classes.extend([obj.label for obj in cell['objects']])
        return list(np.unique(classes))        

    def get_known_words(self):
        words = []
        for d in self.cell_texts:
            words.extend(d.replace('.','').replace(',','').lower().split())
        return list(np.unique(words))

    def collate_fn(batch):
        data = {}
        for k, v in batch[0].items():
            data[k] = [batch[i][k] for i in range(len(batch))]
        return data

class Semantic3dCellRetrievalDatasetMulti(Dataset):
    def __init__(self, path_numpy, path_scenes, scene_names, mention_features, split=None): 
        self.scene_names = scene_names
        self.split = split
        self.datasets = [Semantic3dCellRetrievalDataset(path_numpy, path_scenes, scene_name, mention_features, split) for scene_name in scene_names]

        self.mean_obj = np.mean([dataset.mean_obj for dataset in self.datasets])

        print(str(self))

    def __len__(self):
        return np.sum([len(ds) for ds in self.datasets])

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
        return f'Semantic3dCellRetrievalDatasetMulti: {len(self.scene_names)} scenes, {len(self)} cells, {self.mean_obj: 0.2f} mean objects, split: {self.split}'

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


if __name__ == "__main__":
    dataset1 = Semantic3dPoseReferanceDataset('./data/numpy_merged/', './data/semantic3d', "bildstein_station1_xyz_intensity_rgb", pad_size=8)
    data1 = dataset1[0]
    dataset2 = Semantic3dPoseReferanceMockDataset(6, pad_size=8)
    data2 = dataset2[0]
    quit()

    # dataset = Semantic3dPoseReferanceMockDataset(6, pad_size=8)
    # dataloader = DataLoader(dataset, batch_size=2, collate_fn=Semantic3dObjectReferanceMockDataset.collate_fn)
    # data = dataset[0]
    # batch = next(iter(dataloader))

    #CARE which is from which ;)
    # dataset = Semantic3dCellRetrievalDataset('./data/numpy_merged/', './data/semantic3d', 'bildstein_station1_xyz_intensity_rgb', ['class', 'color', 'position'])
    # data = dataset[0]
    # scene_names = COMBINED_SCENE_NAMES #['bildstein_station1_xyz_intensity_rgb', 'sg27_station5_intensity_rgb']
    
    # dataset_multi_train = Semantic3dCellRetrievalDatasetMulti('./data/numpy_merged/', './data/semantic3d', scene_names, ['class', 'color', 'position'], split='train')
    # dataset_multi_test  = Semantic3dCellRetrievalDatasetMulti('./data/numpy_merged/', './data/semantic3d', scene_names, ['class', 'color', 'position'], split='test')
    # dataloader = DataLoader(dataset_multi_train, batch_size=2, collate_fn=Semantic3dCellRetrievalDataset.collate_fn)
    # batch = next(iter(dataloader))

    # cell_lengths = [len(cell['objects']) for cell in dataset.cells]
    # idx = np.argmax(cell_lengths)
    # img = cv2.flip(draw_cells(dataset.scene_objects, dataset.cells, highlight_idx=idx), 0)
    # cv2.imwrite('cells_dataset.png', img)
    # print(dataset.cell_texts[idx])

    # quit()

    # dataset = Semantic3dObjectReferanceDataset('./data/numpy_merged/', './data/semantic3d', num_distractors=2)
    # dataloader = DataLoader(dataset, batch_size=2, collate_fn=Semantic3dObjectReferanceDataset.collate_fn)
    # data = dataset[0]
    # batch = next(iter(dataloader))

