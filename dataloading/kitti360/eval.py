from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader
from easydict import EasyDict

import torch_geometric.transforms as T 

from datapreparation.kitti360.imports import Object3d, Cell, Pose
from datapreparation.kitti360.drawing import show_pptk, show_objects, plot_cell, plot_pose_in_best_cell
from dataloading.kitti360.poses import batch_object_points
from dataloading.kitti360.base import Kitti360BaseDataset

class Kitti360TopKDataset(Dataset):
    def __init__(self, poses: List[Pose], cells: List[Cell], retrievals, transform, args):
        super().__init__()
        self.poses = poses
        self.retrievals = retrievals
        assert len(poses) == len(retrievals)
        assert len(retrievals[0]) == max(args.top_k), "Retrievals where not trimmed to max(top_k)"
        assert len(poses) != len(cells)
        
        self.cells_dict = {cell.id: cell for cell in cells}
        assert len(self.cells_dict) == len(cells), "Cell-IDs are not unique"

        self.transform = transform
        self.args = args

    def load_pose_and_cell(self, pose: Pose, cell: Cell):
        objects = cell.objects

        # Cut-off objects
        # TODO: Matching possible if best-cell, otherwise just ignore. Pad_size 16 or 32 ok?
        if len(objects) > self.args.pad_size:
            print('Objects overflow: ', len(objects))
            objects = objects[0 : self.args.pad_size]

        while len(objects) < self.args.pad_size:
            objects.append(Object3d.create_padding())

        object_points = batch_object_points(objects, self.transform)

        hint_descriptions = Kitti360BaseDataset.create_hint_description(pose, None)

        return {
            'poses': pose,
            'objects': objects,
            'object_points': object_points,
            'hint_descriptions': hint_descriptions
        }

    def __getitem__(self, idx):
        """Return a "batch" of the pose at idx with each of the corresponding top-k retrieved cells
        """
        pose = self.poses[idx]
        retrievals = self.retrievals[idx]

        return Kitti360TopKDataset.collate_append([
            self.load_pose_and_cell(pose, self.cells_dict[cell_id]) for cell_id in retrievals
        ])

    # NOTE: returns the number of poses, each item has max(top_k) samples!
    def __len__(self):
        return len(self.poses)

    def collate_append(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch    

    def collate_extend(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = []
            for i in range(len(data)):
                assert isinstance(data[i][key], list)
                batch[key].extend(data[i][key])
        return batch

if __name__ == '__main__':
    from dataloading.kitti360.cells import Kitti360CoarseDatasetMulti

    base_path = './data/k360_cs30_cd15_scY_pd10_pc1_spY_closest'
    folder_name = '2013_05_28_drive_0003_sync'    

    args = EasyDict(pad_size=16, top_k=(1, 3, 5))

    transform = T.FixedPoints(256)
    dataset_coarse = Kitti360CoarseDatasetMulti(base_path, [folder_name, ], transform, shuffle_hints=False, flip_poses=False)    

    retrievals = []
    for i in range(len(dataset_coarse.all_poses)):
        retrievals.append([dataset_coarse.all_cells[k].id for k in range(max(args.top_k))])

    dataset = Kitti360TopKDataset(dataset_coarse.all_poses, dataset_coarse.all_cells, retrievals, transform, args)
    data = dataset[0]

    loader = DataLoader(dataset, batch_size=2, collate_fn=Kitti360TopKDataset.collate_append)
    batch = next(iter(loader))