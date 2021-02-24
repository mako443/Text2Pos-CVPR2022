import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch_geometric.nn as gnn

import numpy as np
import os
import os.path as osp
import pickle
import matplotlib.pyplot as plt
from easydict import EasyDict

from models.modules import get_mlp, LanguageEncoder

'''
Dynamic-Graph-based matching modules

TODO:
- use residual connections?
- extract and cat global graph features?
- which aggregation?
- BatchNorm in MLP? (reshape to B*num_obj)
'''

class DGMatch(torch.nn.Module):
    def __init__(self, known_classes, known_words, embedding_dim, num_layers, k, use_residual):
        super(DGMatch, self).__init__()

        self.k = k
        self.use_residual = use_residual
        self.embedding_dim = embedding_dim

        # Set idx=0 for padding/unknown
        self.known_classes = {c: (i+1) for i,c in enumerate(known_classes)}
        self.known_classes['<unk>'] = 0
        self.class_embedding = nn.Embedding(len(self.known_classes), embedding_dim, padding_idx=0)

        self.pos_embedding = get_mlp([2,4,8,16,embedding_dim]) #TODO: optimize these layers

        self.language_encoder = LanguageEncoder(known_words, embedding_dim, bi_dir=True)    

        self.graph_layers = nn.ModuleList([gnn.DynamicEdgeConv(get_mlp([4*embedding_dim, 2*embedding_dim]), k) for i in range(num_layers)])

        d = 2 * embedding_dim if use_residual==False else 2 * num_layers * embedding_dim
        d += embedding_dim
        self.mlp_features = get_mlp([d, 2*embedding_dim])

        #Prediction layers
        self.mlp_object_ref = get_mlp([2*embedding_dim, embedding_dim, 64, 32, 1]) #Predict a reference confidence for each obj
        self.mlp_target_class = get_mlp([embedding_dim, 16, len(self.known_classes)]) #Predict the class of the referred object based on the text
        self.mlp_object_class = get_mlp([2*embedding_dim, embedding_dim, 16, len(self.known_classes)]) #Predict the class of each object based
        self.mlp_object_offset = get_mlp([2*embedding_dim, embedding_dim, 64, 2])

    def forward(self, object_classes, object_positions, descriptions):
        '''
        Encode the descriptions
        '''
        batch_size = len(descriptions)
        description_encodings = self.language_encoder(descriptions) # [B, DIM]
        description_encodings = torch.unsqueeze(description_encodings, dim=1) # [B, 1, DIM]        

        '''
        Encode the objects
        '''    
        num_objects = len(object_classes[0])
        class_indices = torch.zeros((batch_size, num_objects), dtype=torch.long)
        for i in range(batch_size):
            for j in range(num_objects):
                class_indices[i,j] = self.known_classes.get(object_classes[i][j],0)
        class_embeddings = self.class_embedding(class_indices.to(self.device)) # [B, num_obj, DIM]

        pos_embeddings = self.pos_embedding(torch.tensor(object_positions, dtype=torch.float, device=self.device)) # [B, num_obj, DIM]

        object_encodings = class_embeddings + pos_embeddings # [B, num_obj, DIM]

        '''
        Merge object and description encodings (concat the description encoding to every object encoding for combined graph inputs)
        '''
        description_encodings_repeated = description_encodings.repeat(1, num_objects, 1) # [B, num_obj, DIM]
        features = torch.cat((object_encodings, description_encodings_repeated), dim=-1) # [B, num_obj, 2*DIM]        

        '''
        Run graph layers
        '''
        features = features.reshape((-1, 2*self.embedding_dim)) # [B*num_obj, 2*DIM], reshaped to process batch as sparse graph
        batch = torch.zeros(len(features), dtype=torch.long, device=self.device)
        for i in range(batch_size):
            batch[i*num_objects : (i+1)*num_objects] = i

        graph_outputs = []
        for layer in self.graph_layers:
            features = layer(features, batch)
            graph_outputs.append(features)

        if self.use_residual:
            features = torch.cat(graph_outputs, dim=-1)

        features = features.reshape((batch_size, num_objects, -1)) # [B, num_obj, 2*DIM]
        #Concant description again
        features = torch.cat((features, description_encodings_repeated), dim=-1)

        features = self.mlp_features(features) #TODO: use this even w/o use_residual?

        '''
        Make predictions
        '''
        pred_object_ref = torch.squeeze(self.mlp_object_ref(features), dim=-1)
        pred_target_class = torch.squeeze(self.mlp_target_class(description_encodings), dim=1)
        pred_object_class = self.mlp_object_class(features)
        pred_object_offset = self.mlp_object_offset(features)

        model_output = EasyDict()
        model_output.features = features
        model_output.pred_object_ref = pred_object_ref
        model_output.pred_target_class = pred_target_class
        model_output.pred_object_class = pred_object_class
        model_output.pred_object_offset = pred_object_offset
        return model_output

    @property
    def device(self):
        return next(self.pos_embedding.parameters()).device        

if __name__ == "__main__":
    model = DGMatch(['high vegetation', 'low vegetation', 'buildings', 'hard scape', 'cars'], 'a b c d e'.split(), embedding_dim=32, num_layers=2, k=7, use_residual=False)

    object_classes = [['high vegetation', 'low vegetation', 'buildings', 'hard scape', 'cars'] for i in range(3)]
    object_positions = np.random.rand(3, 5, 2)
    descriptions = ['a b x d e', 'a b c x a c', 'a a']

    out = model(object_classes, object_positions, descriptions)
