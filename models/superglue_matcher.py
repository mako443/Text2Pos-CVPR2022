from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Data, Batch

import time
import numpy as np
import os
import pickle
from easydict import EasyDict

from models.modules import get_mlp, LanguageEncoder
from models.superglue import SuperGlue
from models.pointcloud.pointnet2 import PointNet2

from dataloading.semantic3d.semantic3d import Semantic3dObjectReferenceDataset
from dataloading.semantic3d.semantic3d import Semantic3dPoseReferenceMockDataset

from datapreparation.kitti360.imports import Object3d as Object3d_K360

'''
TODO:
- are L2-based distances better?

NOTES:
- BatchNorm yes/no/where? -> Doesn't seem to make much difference
'''

# MLP without trailing ReLU or BatchNorm
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
    def __init__(self, known_classes, known_colors, known_words, args, pointnet_path):
        super(SuperGlueMatch, self).__init__()
        self.embed_dim = args.embed_dim
        self.num_layers = args.num_layers
        self.sinkhorn_iters = args.sinkhorn_iters
        self.use_features = args.use_features
        self.args = args

        self.pointnet = PointNet2(len(known_classes), args) # The known classes are all the same now, at least for K360
        
        load_pn = True
        if load_pn:
            self.pointnet.load_state_dict(torch.load(pointnet_path))
        else:
            print('CARE: Not loading PN state')

        self.pointnet.lin3 = nn.Identity() # Remove the last layer
        self.pointnet_dim = self.pointnet.lin2.weight.size(0)


        self.mlp_object_merge = get_mlp([self.pointnet_dim + self.embed_dim,
                                         max(self.pointnet_dim, self.embed_dim),
                                         self.embed_dim]) # TODO: other variation?

        self.mlp_class = get_mlp([self.pointnet_dim, self.pointnet_dim//2, len(known_classes)])                                        
        self.mlp_color = get_mlp([self.pointnet_dim, self.pointnet_dim//2, len(known_colors)])                                        

        # Set idx=0 for padding
        # self.known_classes = {c: (i+1) for i,c in enumerate(known_classes)}
        # self.known_classes['<unk>'] = 0
        # self.class_embedding = nn.Embedding(len(self.known_classes), self.embed_dim, padding_idx=0)

        self.pos_embedding = get_mlp([3, 64, self.embed_dim], add_batchnorm=True) #OPTION: pos_embedding layers
        # self.color_embedding = get_mlp([3, 64, self.embed_dim], add_batchnorm=True) #OPTION: color_embedding layers
        # self.mlp_merge = get_mlp([len(self.use_features)*self.embed_dim, self.embed_dim], add_batchnorm=True)        

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
        
        print('DEVICE', self.get_device())

    def forward(self, objects, hints, object_points):
        batch_size = len(objects)
        num_objects = len(objects[0])
        '''
        Encode the hints
        '''
        hint_encodings = torch.stack([self.language_encoder(hint_sample) for hint_sample in hints]) # [B, num_hints, DIM]
        hint_encodings = F.normalize(hint_encodings, dim=-1) #Norming those too

        '''
        Get PN++ object features
        '''
        object_features = [self.pointnet(pyg_batch.to(self.get_device())) for pyg_batch in object_points] # [B, pad_size, PN_size]
        object_features = torch.stack(object_features) # [B, pad_size, PN_size]
        object_features = object_features.reshape((batch_size * num_objects, -1))  # [B * pad_size, PN_size]
        object_features = F.normalize(object_features, dim=-1) # [B * pad_size, PN_size]

        positions = [obj.closest_point for objects_sample in objects for obj in objects_sample]
        pos_embedding = self.pos_embedding(torch.tensor(positions, dtype=torch.float, device=self.get_device()))
        pos_embedding = F.normalize(pos_embedding, dim=-1) # [B * pad_size, DIM]

        # Merge and norm
        object_encodings = self.mlp_object_merge(torch.cat((object_features, pos_embedding), dim=-1)) # [B * pad_size, DIM]
        object_encodings = object_encodings.reshape((batch_size, num_objects, self.embed_dim)) # [B, pad_size, DIM]
        object_encodings = F.normalize(object_encodings, dim=-1)

        # Auxiliary predictions
        object_class_preds = self.mlp_class(object_features) # [B * pad_size, num_classes]
        object_color_preds = self.mlp_color(object_features) # [B * pad_size, num_colors]

        '''
        Encode the objects, first flattened for correct batch-norms, then re-shape
        '''
        # num_objects = len(objects[0])
        # class_indices = torch.zeros((batch_size, num_objects), dtype=torch.long, device=self.device)
        # for i in range(batch_size):
        #     for j in range(num_objects):
        #         class_indices[i, j] = self.known_classes.get(objects[i][j].label, 0)

        # embeddings = []
        # if 'class' in self.use_features:
        #     class_embedding = self.class_embedding(torch.tensor(class_indices, dtype=torch.long, device=self.device)).reshape((-1, self.embed_dim))
        #     embeddings.append(F.normalize(class_embedding, dim=-1))

        # if 'color' in self.use_features:
        #     colors = [obj.get_color_rgb() for objects_sample in objects for obj in objects_sample]
        #     color_embedding = self.color_embedding(torch.tensor(colors, dtype=torch.float, device=self.device))
        #     embeddings.append(F.normalize(color_embedding, dim=-1))

        # if 'position' in self.use_features:
        #     positions = [obj.closest_point for objects_sample in objects for obj in objects_sample]
        #     pos_embedding = self.pos_embedding(torch.tensor(positions, dtype=torch.float, device=self.device))
        #     embeddings.append(F.normalize(pos_embedding, dim=-1))

        # if len(embeddings) > 1:
        #     object_encodings = self.mlp_merge(torch.cat(embeddings, dim=-1))
        # else:
        #     object_encodings = embeddings[0]

        # object_encodings = object_encodings.reshape((batch_size, num_objects, self.embed_dim))

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
        outputs.class_preds = object_class_preds
        outputs.color_preds = object_color_preds

        # print("P", outputs.P.shape)

        return outputs

    @property
    def device(self):
        return next(self.pos_embedding.parameters()).device    

    def get_device(self):
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
    args.pointnet_layers = 3
    args.pointnet_variation = 0

    # dataset_train = Semantic3dPoseReferanceMockDataset(args, length=1024)
    # dataloader_train = DataLoader(dataset_train, batch_size=2, collate_fn=Semantic3dPoseReferanceMockDataset.collate_fn)    
    # data = dataset_train[0]
    # batch = next(iter(dataloader_train))

    model = SuperGlueMatch(['class1', 'class2'], ['word1', 'word2'], args, './checkpoints/pointnet_K360.pth')

    # out = model(batch['objects'], batch['hint_descriptions'])

    print('Done')

            


