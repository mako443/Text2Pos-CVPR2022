from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import time
import numpy as np
import os
import pickle
from easydict import EasyDict

from models.modules import get_mlp, LanguageEncoder
from models.superglue import SuperGlue
from dataloading.semantic3d.semantic3d import Semantic3dObjectReferenceDataset
from dataloading.semantic3d.semantic3d import Semantic3dPoseReferenceMockDataset

from datapreparation.kitti360.imports import Object3d as Object3d_K360

'''
TODO:
- BatchNorm yes/no/where?
- are L2-based distances better?
- CARE: when PN++, has model knowlege of object center?
'''

def get_mlp_offset(dims, add_batchnorm=False):
    if len(dims)<3:
        print('get_mlp(): less than 2 layers!')
    mlp = []
    for i in range(len(dims)-1):
        mlp.append(nn.Linear(dims[i], dims[i+1]))
        if i<len(dims)-2:
            mlp.append(nn.ReLU())
            if add_batchnorm:
                mlp.append(nn.BatchNorm1d(dims[i+1]))
    return nn.Sequential(*mlp)

class SuperGlueMatch(torch.nn.Module):
    def __init__(self, known_classes, known_words, args):
        super(SuperGlueMatch, self).__init__()
        self.embed_dim = args.embed_dim
        self.num_layers = args.num_layers
        self.sinkhorn_iters = args.sinkhorn_iters
        self.use_features = args.use_features
        self.args = args

        # Set idx=0 for padding
        self.known_classes = {c: (i+1) for i,c in enumerate(known_classes)}
        self.known_classes['<unk>'] = 0
        self.class_embedding = nn.Embedding(len(self.known_classes), self.embed_dim, padding_idx=0)

        # self.pos_embedding = get_mlp([3,128, self.embed_dim], add_batchnorm=False) #OPTION: pos_embedding layers
        # self.color_embedding = get_mlp([3,128, self.embed_dim], add_batchnorm=False) #OPTION: color_embedding layers
        # self.mlp_merge = get_mlp([3*self.embed_dim, self.embed_dim])

        self.pos_embedding = get_mlp([3, 64, self.embed_dim], add_batchnorm=False) #OPTION: pos_embedding layers
        self.color_embedding = get_mlp([3, 64, self.embed_dim], add_batchnorm=False) #OPTION: color_embedding layers
        self.mlp_merge = get_mlp([len(self.use_features)*self.embed_dim, self.embed_dim], add_batchnorm=False)        

        self.language_encoder = LanguageEncoder(known_words, self.embed_dim, bi_dir=True)  
        self.mlp_offsets = get_mlp_offset([self.embed_dim, self.embed_dim // 2, 2])

        config = {
            'descriptor_dim': self.embed_dim,
            'GNN_layers': ['self', 'cross'] * self.num_layers,
            # 'GNN_layers': ['self', ] * self.num_layers,
            'sinkhorn_iterations': self.sinkhorn_iters,
            'match_threshold': 0.2,
        }
        self.superglue = SuperGlue(config)

    def forward(self, objects, hints):
        batch_size = len(objects)
        '''
        Encode the hints
        '''
        hint_encodings = torch.stack([self.language_encoder(hint_sample) for hint_sample in hints]) # [B, num_hints, DIM]
        hint_encodings = F.normalize(hint_encodings, dim=-1) #Norming those too

        # offsets = self.mlp_offsets(hint_encodings) # [B, num_hints, 2]        

        '''
        Encode the objects, first flattened for correct batch-norms, then re-shape
        '''
        num_objects = len(objects[0])
        class_indices = torch.zeros((batch_size, num_objects), dtype=torch.long, device=self.device)
        for i in range(batch_size):
            for j in range(num_objects):
                class_indices[i, j] = self.known_classes.get(objects[i][j].label, 0)

        embeddings = []
        if 'class' in self.use_features:
            class_embedding = self.class_embedding(torch.tensor(class_indices, dtype=torch.long, device=self.device)).reshape((-1, self.embed_dim))
            embeddings.append(F.normalize(class_embedding, dim=-1))

        if 'color' in self.use_features:
            colors = [obj.get_color_rgb() for objects_sample in objects for obj in objects_sample]
            color_embedding = self.color_embedding(torch.tensor(colors, dtype=torch.float, device=self.device))
            embeddings.append(F.normalize(color_embedding, dim=-1))

        if 'position' in self.use_features:
            positions = [obj.closest_point for objects_sample in objects for obj in objects_sample]
            pos_embedding = self.pos_embedding(torch.tensor(positions, dtype=torch.float, device=self.device))
            embeddings.append(F.normalize(pos_embedding, dim=-1))

        if len(embeddings) > 1:
            object_encodings = self.mlp_merge(torch.cat(embeddings, dim=-1))
        else:
            object_encodings = embeddings[0]

        object_encodings = object_encodings.reshape((batch_size, num_objects, self.embed_dim))

        # class_embeddings = self.class_embedding(class_indices) # [B, num_obj, DIM]
        # if 'class' not in self.use_features:
        #     class_embeddings = torch.zeros_like(class_embeddings)

        # pos_embeddings = torch.zeros((batch_size, num_objects, 3), dtype=torch.float, device=self.device)
        # for i in range(batch_size):
        #     for j in range(num_objects):
        #         if self.args.dataset == 'S3D':
        #             pos_embeddings[i, j, :] = torch.from_numpy(objects[i][j].center)
        #         else:
        #             pos_embeddings[i, j, :] = torch.from_numpy(objects[i][j].closest_point)
        # print(pos_embeddings.shape)
        # print(pos_embeddings[0,0])
        # pos_embeddings = self.pos_embedding(pos_embeddings) # [B, num_obj, DIM]
        # if 'position' not in self.use_features:
        #     pos_embeddings = torch.zeros_like(pos_embeddings)
        
        # color_embeddings = torch.zeros((batch_size, num_objects, 3), dtype=torch.float, device=self.device)
        # for i in range(batch_size):
        #     for j in range(num_objects):
        #         if self.args.dataset == 'S3D':
        #             color_embeddings[i, j, :] = torch.from_numpy(objects[i][j].color)
        #         else:
        #             color_embeddings[i, j, :] = torch.tensor(objects[i][j].get_color_rgb(), dtype=torch.float)
        # color_embeddings = self.color_embedding(color_embeddings) # [B, num_obj, DIM]
        # if 'color' not in self.use_features:
        #     color_embeddings = torch.zeros_like(color_embeddings)

        # object_encodings = self.mlp_merge(torch.cat((class_embeddings, pos_embeddings, color_embeddings), dim=-1)) # [B, num_obj, DIM]
        # object_encodings = F.normalize(object_encodings, dim=-1) # [B, num_obj, DIM]

        '''
        Match object-encodings to hint-encodings
        '''
        desc0 = object_encodings.transpose(1, 2) #[B, DIM, num_obj]
        desc1 = hint_encodings.transpose(1, 2) #[B, DIM, num_hints]
        # print("desc", desc0.shape, desc1.shape)
        
        matcher_output = self.superglue(desc0, desc1)

        '''
        Predict offsets from hints
        '''
        offsets = self.mlp_offsets(hint_encodings) # [B, num_hints, 2]

        outputs = EasyDict()
        outputs.P = matcher_output['P']
        outputs.matches0 = matcher_output['matches0']
        outputs.matches1 = matcher_output['matches1']
        outputs.offsets = offsets

        # print("P", outputs.P.shape)

        return outputs

    @property
    def device(self):
        return next(self.pos_embedding.parameters()).device      

def get_pos_in_cell(objects: List[Object3d_K360], matches0, offsets):
    """Extract a pose estimation relative to the cell (∈ [0,1]²) by
    adding up for each matched objects its location plus offset-vector of corresponding hint,
    then taking the average.

    Args:
        objects (List[Object3d_K360]): List of objects of the cell
        matches0 : matches0 from SuperGlue
        offsets : Offset predictions for each hint

    Returns:
        np.ndarray: Pose estimate
    """
    pose_preds = [] # For each match the object-location plus corresponding offset-vector
    for obj_idx, hint_idx in enumerate(matches0):
        if obj_idx == -1 or hint_idx == -1:
            continue
        pose_preds.append(objects[obj_idx].closest_point[0:2] + offsets[hint_idx]) # Object location plus offset of corresponding hint
    return np.mean(pose_preds, axis=0) if len(pose_preds) > 0 else np.array((0.5,0.5)) # Guess the middle if no matches

if __name__ == "__main__":
    args = EasyDict()
    args.embed_dim = 16
    args.num_layers = 2
    args.sinkhorn_iters = 10
    args.num_mentioned = 4
    args.pad_size = 8
    args.use_features = ['class', 'color', 'position']

    dataset_train = Semantic3dPoseReferanceMockDataset(args, length=1024)
    dataloader_train = DataLoader(dataset_train, batch_size=2, collate_fn=Semantic3dPoseReferanceMockDataset.collate_fn)    
    data = dataset_train[0]
    batch = next(iter(dataloader_train))

    model = SuperGlueMatch(dataset_train.get_known_classes(), dataset_train.get_known_words(), args)   

    out = model(batch['objects'], batch['hint_descriptions'])

    print('Done')

            


