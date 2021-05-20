from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2
from easydict import EasyDict
import time
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T     

from datapreparation.kitti360.utils import CLASS_TO_LABEL, LABEL_TO_CLASS, CLASS_TO_MINPOINTS, CLASS_TO_INDEX
from datapreparation.kitti360.utils import COLORS, COLOR_NAMES, SCENE_NAMES_TRAIN
from datapreparation.kitti360.descriptions import create_synthetic_cell, describe_pose_in_pose_cell, ground_pose_to_best_cell
from datapreparation.kitti360.imports import Object3d, Cell, Pose
from datapreparation.kitti360.drawing import show_pptk, show_objects, plot_objects, plot_cell, plot_pose_in_best_cell
from dataloading.kitti360.base import Kitti360BaseDataset
from dataloading.kitti360.objects import Kitti360ObjectsDatasetMulti
from dataloading.kitti360.poses import load_pose_and_cell


'''
TODO:
- Sample class first or not?
- Explicitly move an object to on-top first or not?
'''

class Kitti360FineSyntheticDataset(Dataset):
    def __init__(self, base_path, scene_names, transform, args, length=1024, fixed_seed=False):
        # Create an objects dataset to copy the objects from in synthetic cell creation
        # CARE: some classes might be empty because of the train/test split
        objects_dataset = Kitti360ObjectsDatasetMulti(base_path, scene_names) # Transform of this dataset is ignored
        self.objects_dict = {c: [] for c in CLASS_TO_INDEX.keys()}
        for obj in objects_dataset.objects:
            self.objects_dict[obj.label].append(obj)

        self.transform = transform
        self.pad_size = args.pad_size
        self.num_mentioned = args.num_mentioned
        self.describe_by = args.describe_by
        self.length = length
        self.fixed_seed = fixed_seed
        self.colors = COLORS
        self.color_names = COLOR_NAMES 

        print(f'Kitti360FineSyntheticDataset, fixed seed: {fixed_seed}, length: {length}, sampling from {len(objects_dataset)} objects, describe {self.describe_by}')    

    def create_synthetic_cell_and_pose(self):
        """Create a synthetic cell for fine localization training.
        Copy ∈ [pad_size, 2*pad_size] objects into ∈ [-0.5, 1.5]^2
        Create pose somewhere in ∈ [0, 1]^2, pose_cell around it (At least 1/4 overlap)
        Create best_cell always [0, 1]^2        

        Returns:
            [type]: [description]
        """
        pose_w = np.random.rand(3)

        # Copy over random objects from the real dataset to random positions
        # Note that the objects are already clustered and normed: taken from cells (not scene) in Kitti360ObjectsDataset

        num_distractors = np.random.randint(self.pad_size - self.num_mentioned) if self.pad_size > self.num_mentioned else 0 # Objects might be deleted later
        cell_objects = []
        for i in range(self.num_mentioned + num_distractors):
            obj_class = np.random.choice([k for k, v in self.objects_dict.items() if len(v) > 0])
            obj = np.random.choice(self.objects_dict[obj_class])
            obj = deepcopy(obj)
            obj.id = i # Set incremental IDs here for later matching.  
            obj.instance_id = i

            obj.xyz[:, 0:2] -= np.mean(obj.xyz[:, 0:2], axis=0) # Shift center to [0, 0]
            obj.xyz[:, 0:2] += np.random.rand(2) # Shift center to random ∈ [0, 1]
            cell_objects.append(obj)                  


        # for i in range(self.num_mentioned + num_distractors):
        #     obj_class = np.random.choice([k for k, v in self.objects_dict.items() if len(v) > 0])
        #     obj = np.random.choice(self.objects_dict[obj_class])
        #     obj.id = i # Set incremental IDs here for later matching.

        #     # Note that an object might be partly outside the cell, but that is ok when masking + clustering is skipped
        #     obj.xyz[:, 0:2] -= np.mean(obj.xyz[:, 0:2], axis=0) # Shift center to [0, 0]
        #     obj.xyz[:, 0:2] += np.random.rand(2) * 2 - 0.5 # Shift center to random ∈ [-0.5, 1.5]^2
        #     cell_objects.append(obj)


        # cell_objects = []
        # for i in range(self.num_mentioned + num_distractors):
        #     obj_class = np.random.choice([k for k, v in self.objects_dict.items() if len(v) > 0])
        #     obj = np.random.choice(self.objects_dict[obj_class])

        #     # Shift the object center to a random position ∈ [0, 1] in x-y-plane, z is kept
        #     # Note that object might be partly outside the cell, but that is ok when masking + clustering is skipped
        #     obj.xyz[:, 0:2] -= np.mean(obj.xyz[:, 0:2], axis=0)
        #     obj.xyz[:, 0:2] += np.random.rand(2)

        #     cell_objects.append(obj)

        # Not really needed anymore
        # Randomly shift an object close to the pose for <on-top>
        # if False: #np.random.choice((True, False)):
        #     idx = np.random.randint(len(cell_objects))
        #     obj = cell_objects[idx]
        #     obj.xyz[:, 0:2] -= np.mean(obj.xyz[:, 0:2], axis=0)
        #     obj.xyz[:, 0:2] += np.array(pose[0:2] + np.random.randn(2)*0.075).reshape((1,2))

        # Create the pose-cell with all objects
        pose_cell = create_synthetic_cell(np.array([0,0,0,1,1,1]), cell_objects)        
        assert pose_cell is not None
        assert np.allclose(pose_cell.cell_size, 1.0)            

        # Describe in pose-cell with all objects
        descriptions = describe_pose_in_pose_cell(pose_w, pose_cell, self.describe_by, self.num_mentioned, max_dist=np.inf) # Use max-dist here since cells have same bbox anyhow (objects explicitly deleted)

        # Randomly delete up to num_mentiond / 2 of the matched objects to create objects-side bins.
        num_delete = np.random.randint(self.num_mentioned / 2 + 1)
        num_delete = min(num_delete, len(cell_objects) - self.num_mentioned) # Don't delete more objects than are needed for valid cell
        matched_ids = [d.object_id for d in descriptions]
        delete_ids = np.random.choice(matched_ids, size=num_delete, replace=False)
        cell_objects = [obj for obj in cell_objects if obj.id not in delete_ids]

        # Describe in best-cell with potentially deleted objects
        best_cell = create_synthetic_cell(np.array([0,0,0,1,1,1]), cell_objects)    
        assert best_cell is not None
        assert np.allclose(best_cell.cell_size, 1.0)  

        descriptions, pose_in_cell, _ = ground_pose_to_best_cell(pose_w, descriptions, best_cell)
        assert np.allclose(pose_in_cell, pose_w)

        pose = Pose(pose_in_cell, pose_w, best_cell.id, descriptions)

        # Debuggings
        if False:
            img = plot_pose_in_best_cell(pose_cell, pose, show_unmatched=True)
            cv2.imshow("pose", img)
            cv2.waitKey()          

        if False:
            print('Num del:', num_delete)
            for d in descriptions:
                print(d)

            img0 = plot_pose_in_best_cell(pose_cell, pose, show_unmatched=True)
            img1 = plot_pose_in_best_cell(best_cell, pose)
            cv2.imshow("pose", img0)
            cv2.imshow("best", img1)
            cv2.waitKey()

        # # Create the cell
        # cell = create_cell(-1, "MOCK", np.array([0,0,0,1,1,1]), cell_objects, is_synthetic=True)
        # assert cell is not None

        # Create the pose
        # pose = describe_pose(pose, cell)

        return best_cell, pose

    def __getitem__(self, idx):
        """Return the data of a synthetic cell.
        """
        if self.fixed_seed:
            np.random.seed(idx)
        t0 = time.time()

        cell, pose = self.create_synthetic_cell_and_pose()
        hints = Kitti360BaseDataset.create_hint_description(pose, cell)

        return load_pose_and_cell(pose, cell, hints, self.pad_size, self.transform)

    def __len__(self):
        return self.length

    def collate_fn(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch      

    def get_known_classes(self):
        return list(self.objects_dict.keys())

    def get_known_words(self):
        known_words = []
        for i in range(50):
            data = self[i]
            for hint in data['hint_descriptions']:
                known_words.extend(hint.replace('.','').replace(',','').lower().split())
        return list(np.unique(known_words))  

if __name__ == '__main__':
    base_path = './data/k360_cs30_cd15_scY_pd10_pc1_spY'
    folder_name = '2013_05_28_drive_0003_sync'
    args = EasyDict(pad_size=16, num_mentioned=6)

    transform = T.Compose([T.FixedPoints(1024), T.NormalizeScale()])
    dataset = Kitti360FineSyntheticDataset(base_path, [folder_name, ], transform, args)
    data = dataset[0]
    
