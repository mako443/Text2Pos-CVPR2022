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

        self.language_encoder = LanguageEncoder(known_words, embed_dim, bi_dir=True)  

        config = {
            'descriptor_dim': embed_dim,
            'GNN_layers': ['self', 'cross'] * num_layers,
            'sinkhorn_iterations': sinkhorn_iters,
            'match_threshold': 0.2,
        }
        self.superglue = SuperGlue(config)

    #Currently not batches!
    def forward(self, object_classes, object_positions, hints):
        '''
        Encode the hints
        '''
        hint_encodings = self.language_encoder(hints) # [num_hints, DIM]

        '''
        Encode the objects
        '''    
        num_objects = len(object_classes)
        class_indices = torch.zeros(num_objects, dtype=torch.long)
        for j in range(num_objects):
            class_indices[j] = self.known_classes.get(object_classes[j],0)
        class_embeddings = self.class_embedding(class_indices.to(self.device)) # [num_obj, DIM]

        pos_embeddings = self.pos_embedding(torch.tensor(object_positions, dtype=torch.float, device=self.device)) # [num_obj, DIM]

        object_encodings = F.normalize(class_embeddings) + F.normalize(pos_embeddings) # [num_obj, DIM], normalize for equal magnitudes
        object_encodings = F.normalize(object_encodings) #Normalize for stable matching

        '''
        Match object-encodings to hint-encodings
        '''
        desc0 = object_encodings.transpose(0,1).unsqueeze(dim=0) #[1, DIM, num_obj]
        desc1 = hint_encodings.transpose(0,1).unsqueeze(dim=0) #[1, DIM, num_hints]
        
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
    model = SuperGlueMatch(['high vegetation', 'low vegetation', 'buildings', 'hard scape', 'cars'], 'a b c d e'.split(), 300)

    out = model(['high vegetation', 'low vegetation', 'buildings', 'hard scape', 'cars', 'xx'], np.random.rand(6,2), ['a b x d e', 'a b c x a c', 'a a'])

