import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time
import numpy as np
import os
import pickle
from easydict import EasyDict

from models.modules import get_mlp, LanguageEncoder
from models.superglue import SuperGlue

'''
TODO:
- optimize SuperGlue params
- CARE: norm embeddings before SuperGlue
- implement batching
'''

class SuperGlueMatch(torch.nn.Module):
    def __init__(self, known_classes, known_words, embed_dim, num_layers, sinkhorn_iters):
        super(SuperGlueMatch, self).__init__()

        # Set idx=0 for padding
        self.known_classes = {c: (i+1) for i,c in enumerate(known_classes)}
        self.known_classes['<unk>'] = 0
        self.class_embedding = nn.Embedding(len(self.known_classes), embed_dim, padding_idx=0)

        self.pos_embedding = get_mlp([2,128, embed_dim]) #OPTION: pos_embedding layers
        self.color_embedding = get_mlp([3,128, embed_dim]) #OPTION: color_embedding layers

        self.language_encoder = LanguageEncoder(known_words, embed_dim, bi_dir=True)  

        config = {
            'descriptor_dim': embed_dim,
            'GNN_layers': ['self', 'cross'] * num_layers,
            'sinkhorn_iterations': sinkhorn_iters,
            'match_threshold': 0.2,
        }
        self.superglue = SuperGlue(config)

    #Currently not batches!
    def forward(self, object_classes, object_positions, hints, object_colors=None):
        batch_size = len(object_classes)
        '''
        Encode the hints
        '''
        hint_encodings = torch.stack([self.language_encoder(hint_sample) for hint_sample in hints]) # [B, num_hints, DIM]

        '''
        Encode the objects
        '''    
        num_objects = len(object_classes[0])
        class_indices = torch.zeros((batch_size, num_objects), dtype=torch.long)
        for i in range(batch_size):
            for j in range(num_objects):
                class_indices[i, j] = self.known_classes.get(object_classes[i][j],0)
        class_embeddings = self.class_embedding(class_indices.to(self.device)) # [B, num_obj, DIM]

        pos_embeddings = self.pos_embedding(torch.tensor(object_positions, dtype=torch.float, device=self.device)) # [B, num_obj, DIM]
        object_encodings = F.normalize(class_embeddings, dim=-1) + F.normalize(pos_embeddings, dim=-1) # [B, num_obj, DIM], normalize for equal magnitudes
        
        if object_colors is not None:
            color_embeddings = self.color_embedding(torch.tensor(object_colors, dtype=torch.float, device=self.device)) # [B, num_obj, DIM]
            object_encodings += F.normalize(color_embeddings, dim=-1)

        object_encodings = F.normalize(object_encodings, dim=-1) # [B, num_obj, DIM] Normalize for stable matching

        '''
        Match object-encodings to hint-encodings
        '''
        desc0 = object_encodings.transpose(1, 2) #[B, DIM, num_obj]
        desc1 = hint_encodings.transpose(1, 2) #[B, DIM, num_hints]
        
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
    model = SuperGlueMatch(['high vegetation', 'low vegetation', 'buildings', 'hard scape', 'cars'], 'a b c d e'.split(), 300, num_layers=2, sinkhorn_iters=10)

    batch_size = 3
    out = model([['high vegetation', 'low vegetation', 'buildings', 'hard scape', 'cars', 'xx'] for _ in range(batch_size)], 
                np.random.rand(batch_size, 6, 2), 
                [['a b x d e', 'a b c x a c', 'a a'] for _ in range(batch_size)])

    print('Done')

            


