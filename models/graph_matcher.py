import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric.nn as gnn

import time
import numpy as np
import os
import pickle
from easydict import EasyDict

from models.modules import get_mlp, LanguageEncoder
from models.superglue import SuperGlue

from dataloading.semantic3d import Semantic3dObjectReferanceDataset

'''
Graph-based models for explicit-matching (objects <-> hints)

TODO:
- extract and cat global graph features? (see paper again)
- DGCNN or GraphAttn?
- why the limited capacity? why multi-layer so bad?
'''

class GraphMatch(torch.nn.Module):
    def __init__(self, known_classes, known_words, embed_dim, k, sinkhorn_iters, num_layers, use_features=['class', 'color', 'position']):
        super(GraphMatch, self).__init__()
        self.embed_dim = embed_dim
        self.use_features = use_features
        self.num_layers = num_layers
        assert num_layers in (0,1,2)

        '''
        Object path
        '''        
        # Set idx=0 for padding
        self.known_classes = {c: (i+1) for i,c in enumerate(known_classes)}
        self.known_classes['<unk>'] = 0
        self.class_embedding = nn.Embedding(len(self.known_classes), embed_dim, padding_idx=0)

        self.pos_embedding = get_mlp([3,128, embed_dim]) #OPTION: pos_embedding layers
        self.color_embedding = get_mlp([3,128, embed_dim]) #OPTION: color_embedding layers        

        self.mlp_merge = get_mlp([len(use_features)*embed_dim, embed_dim])

        self.graph1 = gnn.DynamicEdgeConv(get_mlp([2 * embed_dim, embed_dim, embed_dim], add_batchnorm=True), k=k, aggr='max')
        self.graph2 = gnn.DynamicEdgeConv(get_mlp([2 * embed_dim, embed_dim, embed_dim], add_batchnorm=True), k=k, aggr='max')
        # self.graph_obj2 = gnn.DynamicEdgeConv(get_mlp([2 * embed_dim, embed_dim, embed_dim], add_batchnorm=True), k=k, aggr='max')
        self.mlp_residual = get_mlp([2*embed_dim, embed_dim, embed_dim])

        #Using separate Graph networks does not seem to help...
        # self.graph_txt1 = gnn.DynamicEdgeConv(get_mlp([2 * embed_dim, embed_dim, embed_dim], add_batchnorm=True), k=k, aggr='max')
        # self.graph_txt2 = gnn.DynamicEdgeConv(get_mlp([2 * embed_dim, embed_dim, embed_dim], add_batchnorm=True), k=k, aggr='max')
        # self.mlp_residual_txt = get_mlp([2*embed_dim, embed_dim, embed_dim])

        self.mlp_obj = get_mlp([embed_dim, embed_dim, embed_dim], add_batchnorm=True)
        self.mlp_txt = get_mlp([embed_dim, embed_dim, embed_dim], add_batchnorm=True)

        '''
        Textual path
        '''
        self.language_encoder = LanguageEncoder(known_words, embed_dim, bi_dir=True)                      

        '''
        Matcher (SuperGlue w/ empty GNN, i.e. just the Sinkhorn layer)
        '''
        config = {
            'descriptor_dim': embed_dim,
            'GNN_layers': [],
            'sinkhorn_iterations': sinkhorn_iters,
            'match_threshold': 0.2,
        }
        self.superglue = SuperGlue(config)        

        print(f'GraphMatch: dim {embed_dim}, k {k}, features {self.use_features}')

    def forward(self, objects, hints):
        batch_size = len(objects)
        '''
        Encode the hints
        '''
        # hint_encodings = torch.stack([self.language_encoder(hint_sample) for hint_sample in hints]) # [B, num_hints, DIM]
        # hint_encodings = F.normalize(hint_encodings, dim=-1) # [B, num_hints, DIM], norming all SuperGlue inputs
        hints_flat = []
        batch_hints = [] #Batch tensor to send into PyG 
        for i_batch, hints_sample in enumerate(hints):
            for hint in hints_sample:
                hints_flat.append(hint)
                batch_hints.append(i_batch)
        hint_encodings = self.language_encoder(hints_flat) # [batch_size * num_hints, DIM]
        batch_hints = torch.tensor(batch_hints, dtype=torch.long, device=self.device)  

        '''
        Process the objects in a flattened way first
        '''
        class_indices = []
        batch_obj = [] #Batch tensor to send into PyG
        for i_batch, objects_sample in enumerate(objects):
            for obj in objects_sample:
                class_idx = self.known_classes.get(obj.label, 0)
                class_indices.append(class_idx)
                batch_obj.append(i_batch)
        batch_obj = torch.tensor(batch_obj, dtype=torch.long, device=self.device) 

        embeddings = []
        if 'class' in self.use_features:
            class_embedding = self.class_embedding(torch.tensor(class_indices, dtype=torch.long, device=self.device))
            embeddings.append(F.normalize(class_embedding, dim=-1))
        if 'color' in self.use_features:
            colors = []
            for objects_sample in objects:
                colors.extend([obj.color for obj in objects_sample])
            color_embedding = self.color_embedding(torch.tensor(colors, dtype=torch.float, device=self.device))
            embeddings.append(F.normalize(color_embedding, dim=-1))
        if 'position' in self.use_features:
            positions = []
            for objects_sample in objects:
                positions.extend([obj.center for obj in objects_sample])
            pos_embedding = self.pos_embedding(torch.tensor(positions, dtype=torch.float, device=self.device))
            embeddings.append(F.normalize(pos_embedding, dim=-1))

        # object_encodings ∈ [batch_size * num_obj, DIM], i.e. flattened batch for PyG
        if len(embeddings) > 1:
            obj_encodings = self.mlp_merge(torch.cat(embeddings, dim=-1))
        else:
            obj_encodings = embeddings[0]

        '''
        Run graph on both sides
        '''
        if self.num_layers == 0: #Only MLP for each side
            obj_encodings = self.mlp_obj(obj_encodings)
            hint_encodings = self.mlp_obj(hint_encodings)
        elif self.num_layers == 1:
            obj_encodings = self.graph1(obj_encodings, batch_obj)
            hint_encodings = self.graph1(hint_encodings, batch_hints)
        else:
            obj_encodings1 = self.graph1(obj_encodings, batch_obj)
            obj_encodings2 = self.graph2(obj_encodings1, batch_obj)
            a = torch.cat((obj_encodings1, obj_encodings2), dim=-1)
            obj_encodings = self.mlp_residual(torch.cat((obj_encodings1, obj_encodings2), dim=-1))

            hint_encodings1 = self.graph1(hint_encodings, batch_hints)
            hint_encodings2 = self.graph2(hint_encodings1, batch_hints)
            hint_encodings = self.mlp_residual(torch.cat((hint_encodings1, hint_encodings2), dim=-1))


        '''
        Run matcher
        '''
        #CARE: which embeddings are fed in!
        obj_encodings = F.normalize(obj_encodings.reshape((batch_size, -1, self.embed_dim)), dim=-1) # [B, num_obj, DIM]
        hint_encodings = F.normalize(hint_encodings.reshape((batch_size, -1, self.embed_dim)), dim=-1) # [B, num_hints, DIM]

        desc0 = obj_encodings.transpose(1,2) # [B, DIM, num_obj]
        desc1 = hint_encodings.transpose(1,2) # [B, DIM, num_hints]

        matcher_output = self.superglue(desc0, desc1)

        outputs = EasyDict()
        outputs.P = matcher_output['P']
        outputs.matches0 = matcher_output['matches0']
        outputs.matches1 = matcher_output['matches1']

        return outputs
        
        

    @property
    def device(self):
        return next(self.pos_embedding.parameters()).device             


if __name__ == "__main__":
    model = GraphMatch(['high vegetation', 'low vegetation', 'buildings', 'hard scape', 'cars'], 'a b c d e'.split(), 16, 2, sinkhorn_iters=10)

    dataset = Semantic3dObjectReferanceDataset('./data/numpy_merged/', './data/semantic3d', num_distractors=2)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=Semantic3dObjectReferanceDataset.collate_fn)
    data = dataset[0]
    batch = next(iter(dataloader))

    out = model.forward(batch['objects'], batch['hint_descriptions'])

