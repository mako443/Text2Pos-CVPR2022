import os
import os.path as osp
import sys
import time
import h5py
import json
import pickle
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

from datapreparation.imports import Object3D, DescriptionObject, Pose, COMBINED_SCENE_NAMES
from datapreparation.descriptions import describe_cell
from datapreparation.prepare_semantic3d import create_cells
from datapreparation.drawing import draw_cells, draw_retrieval

class Semantic3dPosesDataset(Dataset):
    def __init__(self, path_numpy, path_scenes, scene_name, cell_size, cell_stride, split=None):
        self.path_numpy = path_numpy
        self.path_scenes = path_scenes
        self.split = split
        self.cell_size = cell_size
        self.cell_stride = cell_stride
    
        self.scene_name = scene_name #'sg27_station5_intensity_rgb'

        #Load objects
        self.scene_objects = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'objects.pkl'), 'rb'))
        self.poses = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'poses.pkl'), 'rb'))
        self.pose_descriptions = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'pose_descriptions.pkl'), 'rb'))
        assert len(self.poses)==len(self.pose_descriptions)

        #Apply split: select poses, but keep all cells
        if split is not None:
            assert split in ('train', 'test')
            test_indices = (np.arange(len(self.poses)) % 5) == 0
            indices = test_indices if split=='test' else np.bitwise_not(test_indices)
            self.poses = [p for (i,p) in enumerate(self.poses) if indices[i]]
            self.pose_descriptions = [pd for (i, pd) in enumerate(self.pose_descriptions) if indices[i]]         

        #Create texts
        self.pose_texts = [self.create_pose_text(description) for description in self.pose_descriptions]

        #Create cells
        self.cells, _ = create_cells(self.scene_objects, cell_size=self.cell_size, cell_stride=self.cell_stride)
        self.best_cell_indices = [self.find_best_cell(pose) for pose in self.poses]     

        assert len(self.poses) == len(self.pose_descriptions) == len(self.pose_texts) == len(self.best_cell_indices)

        print(self)

    # CARE: Averaged among the best cells, not all cells!
    def gather_stats(self):
        center_dists = []
        fractions_objects_in_cell = []
        distractors_in_cell = []
        for i_pose, (pose, pose_description) in enumerate(zip(self.poses, self.pose_descriptions)):
            cell = self.cells[self.best_cell_indices[i_pose]]
            center_dists.append(np.linalg.norm(pose.eye[0:2] - cell.center) / self.cell_size)
            
            ids_in_cell = [obj.id for obj in cell.objects]
            num_pose_objects_in_cell = np.sum([obj.id in ids_in_cell for obj in pose_description])
            num_distractor_objects_in_cell = len(ids_in_cell) - num_pose_objects_in_cell

            fractions_objects_in_cell.append(num_pose_objects_in_cell / len(pose_description))
            distractors_in_cell.append(num_distractor_objects_in_cell)

        return {
            "center_distances": np.mean(center_dists),
            "fractions_in_cell": np.mean(fractions_objects_in_cell),
            "distractors_in_cell": np.mean(distractors_in_cell)
        }


    def find_best_cell(self, pose):
        dists = [cell.center - pose.eye[0:2] for cell in self.cells]
        dists = np.linalg.norm(dists, axis=1)
        return np.argmin(dists)

    def __repr__(self):
        return f'Semantic3dPosesDataset ({self.scene_name}), {len(self.poses)} poses, {len(self.scene_objects)} objects, {len(self.cells)} cells'
    
    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        pose = self.poses[idx]
        text = self.pose_texts[idx]
        cell_idx = self.best_cell_indices[idx]
        cell = self.cells[cell_idx]

        negative_cell_idx = np.random.choice([i for i in range(len(self.cells)) if i!=cell_idx])
        negative_cell = self.cells[negative_cell_idx]

        return {
            'poses': pose,
            'texts': text,
            'cells': cell,
            'cell_indices': cell_idx,
            'negative_cells': negative_cell
        }

    def collate_fn(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch

    def create_pose_text(self, description):
        text = "The pose is "
        for i, do in enumerate(description):
            text += f'{do.direction} of a {do.color_text} {do.label}'
            if i < len(description)-1:
                text += " and "
        text += "."

        return text

    def plot(self, pose_indices='all'):
        if pose_indices=='all': pose_indices = np.arange(len(self))

        highlight_indices = [self.best_cell_indices[idx] for idx in pose_indices]
        poses = [self.poses[i] for i in pose_indices]
        pose_descriptions = [self.pose_descriptions[i] for i in pose_indices]
        img = cv2.flip(draw_cells(self.scene_objects, self.cells, poses=poses, pose_descriptions=pose_descriptions, highlight_indices=highlight_indices), 0)
        return img

    def get_known_classes(self):
        return list(np.unique([obj.label for obj in self.scene_objects]))

    def get_known_words(self):
        words = []
        for d in self.pose_texts:
            words.extend(d.replace('.','').replace(',','').lower().split())
        return list(np.unique(words))

class Semantic3dPosesDatasetMulti(Dataset):
    def __init__(self, path_numpy, path_scenes, scene_names, cell_size, cell_stride, split=None):
        self.scene_names = scene_names
        self.split = split
        self.datasets = [Semantic3dPosesDataset(path_numpy, path_scenes, scene_name, cell_size, cell_stride, split=split) for scene_name in scene_names]
        self.cells = []
        for ds in self.datasets:
            self.cells.extend(ds.cells)

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
        return (
            f'Semantic3dPosesDatasetMulti ({len(self.scene_names)} scenes), '
            f'{len(self)} poses, '
            f'{np.sum([len(ds.scene_objects) for ds in self.datasets])} objects, '
            f'{np.sum([len(ds.cells) for ds in self.datasets])} cells'
        )
    
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

    def gather_stats(self):
        all_stats = [ds.gather_stats() for ds in self.datasets]
        return { k: np.mean([stats[k] for stats in all_stats]) for k in all_stats[0].keys()}

if __name__ == '__main__':
    dataset = Semantic3dPosesDataset('./data/numpy_merged/', './data/semantic3d', "sg27_station1_intensity_rgb", 60, 40, split='test')
    img = draw_retrieval(dataset, 0, [0,1,2])
    cv2.imshow("", img)
    cv2.waitKey()
    quit()

    dataset = Semantic3dPosesDatasetMulti('./data/numpy_merged/', './data/semantic3d', COMBINED_SCENE_NAMES, cell_size=65.0, cell_stride=65.0/3)
    print(dataset.gather_stats())
    quit()

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=Semantic3dPosesDataset.collate_fn)
    data = dataset[0]        
    batch = next(iter(dataloader))
    print(dataset.gather_stats())
    quit()

    for idx in range(len(dataset)):
        img = dataset.plot([idx,])
        cv2.imshow("", img); k = cv2.waitKey()
        if k==ord('q'): break