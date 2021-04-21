import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

# def get_mlp(dims, add_batchnorm=False):
#     if len(dims)<3:
#         print('get_mlp(): less than 2 layers!')
#     mlp = []
#     for i in range(len(dims)-1):
#         mlp.append(nn.Linear(dims[i], dims[i+1]))
#         if i<len(dims)-2:
#             mlp.append(nn.ReLU())
#             if add_batchnorm:
#                 mlp.append(nn.BatchNorm1d(dims[i+1]))
#     return nn.Sequential(*mlp)

# CARE: This has a trailing ReLU!!
def get_mlp(channels, add_batchnorm=True):
    if add_batchnorm:
        return nn.Sequential(*[
            nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i]), nn.ReLU())
            for i in range(1, len(channels))
        ])
    else:
        return nn.Sequential(*[
            nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU())
            for i in range(1, len(channels))
        ])    


class LanguageEncoder(torch.nn.Module):
    def __init__(self, known_words, embedding_dim, bi_dir, num_layers=1):
        super(LanguageEncoder, self).__init__()

        self.known_words = {c: (i+1) for i,c in enumerate(known_words)}
        self.known_words['<unk>'] = 0        
        self.word_embedding = nn.Embedding(len(self.known_words), embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, bidirectional=bi_dir, num_layers=num_layers)

    '''
    Encodes descriptions as batch [d1, d2, d3, ..., d_B] with d_i a string. Strings can be of different sizes.
    '''
    def forward(self, descriptions):
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

        d = 2 * self.lstm.num_layers if self.lstm.bidirectional else 1 * self.lstm.num_layers
        h=torch.zeros(d, batch_size, self.word_embedding.embedding_dim).to(self.device)
        c=torch.zeros(d, batch_size, self.word_embedding.embedding_dim).to(self.device)

        _,(h,c) = self.lstm(description_inputs, (h,c))
        description_encodings = torch.mean(h, dim=0) # [B, DIM] TODO: cat even better?

        return description_encodings

    @property
    def device(self):
        return next(self.lstm.parameters()).device        