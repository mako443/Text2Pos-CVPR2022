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

'''
TODO:
- are L2-based distances better?
- CARE: when PN++, has model knowlege of object center?
'''

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

        self.pos_embedding = get_mlp([3,128, self.embed_dim]) #OPTION: pos_embedding layers
        self.color_embedding = get_mlp([3,128, self.embed_dim]) #OPTION: color_embedding layers

        self.mlp_merge = get_mlp([3*self.embed_dim, self.embed_dim])

        self.language_encoder = LanguageEncoder(known_words, self.embed_dim, bi_dir=True)  
        self.mlp_offsets = get_mlp([self.embed_dim, self.embed_dim //2, 2])

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

        '''
        Encode the objects
        '''    
        num_objects = len(objects[0])
        class_indices = torch.zeros((batch_size, num_objects), dtype=torch.long, device=self.device)
        for i in range(batch_size):
            for j in range(num_objects):
                class_indices[i, j] = self.known_classes.get(objects[i][j].label, 0)
        class_embeddings = self.class_embedding(class_indices) # [B, num_obj, DIM]
        if 'class' not in self.use_features:
            class_embeddings = torch.zeros_like(class_embeddings)

        pos_embeddings = torch.zeros((batch_size, num_objects, 3), dtype=torch.float, device=self.device)
        for i in range(batch_size):
            for j in range(num_objects):
                if self.args.dataset == 'S3D':
                    pos_embeddings[i, j, :] = torch.from_numpy(objects[i][j].center)
                else:
                    pos_embeddings[i, j, :] = torch.from_numpy(objects[i][j].closest_point)
        pos_embeddings = self.pos_embedding(pos_embeddings) # [B, num_obj, DIM]
        if 'position' not in self.use_features:
            pos_embeddings = torch.zeros_like(pos_embeddings)
        
        color_embeddings = torch.zeros((batch_size, num_objects, 3), dtype=torch.float, device=self.device)
        for i in range(batch_size):
            for j in range(num_objects):
                if self.args.dataset == 'S3D':
                    color_embeddings[i, j, :] = torch.from_numpy(objects[i][j].color)
                else:
                    color_embeddings[i, j, :] = torch.tensor(objects[i][j].get_color_rgb(), dtype=torch.float)
        color_embeddings = self.color_embedding(color_embeddings) # [B, num_obj, DIM]
        if 'color' not in self.use_features:
            color_embeddings = torch.zeros_like(color_embeddings)

        object_encodings = self.mlp_merge(torch.cat((class_embeddings, pos_embeddings, color_embeddings), dim=-1)) # [B, num_obj, DIM]
        object_encodings = F.normalize(object_encodings, dim=-1) # [B, num_obj, DIM]

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
        # offsets = F.normalize(offsets, dim=-1)

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

            


