import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import os
import pickle

'''
Matching Modules

TODO:
- why is "wrong" order better??
- encode obj color
- encode position or full bboxes?
'''

def get_mlp(dims):
    # mlp = nn.Sequential(*[nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
    mlp = []
    for i in range(len(dims)-1):
        mlp.append(nn.Linear(dims[i], dims[i+1]))
        if i<len(dims)-2:
            mlp.append(nn.ReLU())
    return nn.Sequential(*mlp)

class TransformerMatch1(torch.nn.Module):
    def __init__(self, known_classes, known_words, embedding_dim, num_layers):
        super(TransformerMatch1, self).__init__()

        # Set idx=0 for padding
        self.known_classes = {c: (i+1) for i,c in enumerate(known_classes)}
        self.known_classes['<unk>'] = 0
        self.class_embedding = nn.Embedding(len(self.known_classes), embedding_dim, padding_idx=0)

        self.pos_embedding = get_mlp([2,4,8,16,embedding_dim]) #TODO: optimize these layers

        # TODO: switch to Glove?
        self.known_words = {c: (i+1) for i,c in enumerate(known_words)}
        self.known_words['<unk>'] = 0        
        self.word_embedding = nn.Embedding(len(self.known_words), embedding_dim, padding_idx=0)

        #Self attention layers
        self.encoder_layers = nn.ModuleList([nn.TransformerEncoderLayer(embedding_dim + embedding_dim, nhead=8, dim_feedforward=2048)])
        #TODO: do Xavier?
        for layer in self.encoder_layers:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)        

        #Text encoding
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, bidirectional=False)

        # TODO: One MLP for all or separate after each TF layer?
        #self.mlp_ref_type = get_mlp([2*embedding_dim, 256, 128, 64, 32, 3]) #TODO: num aux.-classes (2 or 3), optimize MLP
        self.mlp_ref_type = get_mlp([2*embedding_dim, embedding_dim, 1])

        self.mlp_target_class = get_mlp([embedding_dim, 16, len(self.known_classes)])

        self.mlp_object_class = get_mlp([2*embedding_dim, embedding_dim, 16, len(self.known_classes)])

    #TODO: if objects are the same, feed them in once or as full batch? -> Probably as batch for separate aux.-losses
    '''
    Objects as [obj1, obj2, ...], assumed to be the same for all descriptions
    Descriptions as batch [d1, d2, d3, ..., d_B] with d_i a string. Strings can be of different sizes.
    ## Object referance types [B, num_obj ∈ {0,1,2}: ground-truth whether each object is unrelated, mentioned, or the target. Used for aux.-loss.
    '''
    def forward(self, object_classes, object_positions, descriptions):
        '''
        Encode the descriptions
        '''
        word_indices = [ [self.known_words.get(word, 0) for word in description.replace('.', '').replace(',', '').lower().split()] for description in descriptions]
        description_lengths = [len(w) for w in word_indices]
        batch_size, max_length = len(word_indices), max(description_lengths)
        padded_indices = np.zeros((batch_size,max_length), np.int)

        for i,caption_length in enumerate(description_lengths):
            padded_indices[i,:caption_length] = word_indices[i]
        
        padded_indices = torch.from_numpy(padded_indices)
        padded_indices = padded_indices.to(self.device) #Possibly move to cuda

        embedded_words = self.word_embedding(padded_indices)
        description_inputs = nn.utils.rnn.pack_padded_sequence(embedded_words, torch.tensor(description_lengths), batch_first=True, enforce_sorted=False)   

        h=torch.zeros(1,batch_size,self.word_embedding.embedding_dim).to(self.device)
        c=torch.zeros(1,batch_size,self.word_embedding.embedding_dim).to(self.device)

        _,(h,c) = self.lstm(description_inputs, (h,c))
        description_encodings = torch.squeeze(h, dim=0) # [B, DIM]
        # description_encodings = torch.mean(h, dim=0)
        description_encodings = torch.unsqueeze(description_encodings, dim=1) # [B, 1, DIM]

        # description_encodings = F.relu(description_encodings) #TODO: do this or not?

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

        #TODO: norm somewhere?
        '''
        Merge object and description encodings (concat the description encoding to every object encoding for combined transformer inputs)
        '''
        description_encodings_repeated = description_encodings.repeat(1, num_objects, 1) # [B, num_obj, DIM]
        transformer_input = torch.cat((object_encodings, description_encodings_repeated), dim=-1) # [B, num_obj, 2*DIM]

        '''
        Run Tranformer Encoder Layers
        '''
        # features = torch.transpose(transformer_input, 0, 1) # [B, num_obj, E] -> [num_obj, B, E]
        features = transformer_input #TODO: transpose or not?!
        for layer in self.encoder_layers:
            features = layer(features)

        '''
        Make predictions
        '''
        obj_ref_predictions = self.mlp_ref_type(features) # [num_obj, B, 1]
        # obj_ref_predictions = torch.transpose(torch.squeeze(obj_ref_predictions, dim=-1), 0, 1) # [num_obj, B, 1] -> [B, num_obj]
        obj_ref_predictions = torch.squeeze(obj_ref_predictions, dim=-1)

        target_class_pred = torch.squeeze(self.mlp_target_class(description_encodings), dim=1)

        obj_class_pred = self.mlp_object_class(features)

        return features, obj_ref_predictions, target_class_pred, obj_class_pred


    @property
    def device(self):
        return next(self.lstm.parameters()).device


if __name__ == "__main__":
    model = TransformerMatch1(['high vegetation', 'low vegetation', 'buildings', 'hard scape', 'cars'], 'a b c d e'.split(), 300, 2)

    model([['high vegetation', 'low vegetation', 'buildings', 'hard scape', 'cars', 'xx'] for i in range(3)], ['a b x d e', 'a b c x a c', 'a a'])
