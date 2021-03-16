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
from datapreparation.drawing import draw_cells

class Semantic3dPosesDataset(Dataset):
    def __init__(self, path_numpy, path_scenes, num_distractors='all', split=None):
        self.path_numpy = path_numpy
        self.path_scenes = path_scenes
        self.num_distractors = num_distractors
        self.split = split
    
        self.scene_name = 'sg27_station5_intensity_rgb'

        #Load objects
        self.scene_objects = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'objects.pkl'), 'rb'))
        self.poses = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'poses.pkl'), 'rb'))
        self.pose_descriptions = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'pose_descriptions.pkl'), 'rb'))
        assert len(self.poses)==len(self.pose_descriptions)

        #Create texts
        self.pose_texts = [self.create_pose_text(description) for description in self.pose_descriptions]

        #Create cells
        self.cells, _ = create_cells(self.scene_objects, cell_size=65, cell_stride=65/3)
        self.best_cell_indices = [self.find_best_cell(pose) for pose in self.poses]

        print(self)

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

        return {
            'poses': pose,
            'texts': text,
            'cells': cell,
            'cell_indices': cell_idx
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

if __name__ == '__main__':
    dataset = Semantic3dPosesDataset('./data/numpy_merged/', './data/semantic3d')
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=Semantic3dPosesDataset.collate_fn)
    data = dataset[0]        
    batch = next(iter(dataloader))
    quit()

    for idx in range(len(dataset)):
        img = dataset.plot([idx,])
        cv2.imshow("", img); k = cv2.waitKey()
        if k==ord('q'): break