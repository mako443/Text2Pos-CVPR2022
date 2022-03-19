import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import os
import pickle


class TextPoseNet(torch.nn.Module):
    def __init__(self, known_words, embedding_dim):
        super(TextPoseNet, self).__init__()

        self.embedding_dim = embedding_dim

        self.padding_idx = 0  # Padding idx is also used for unkown words (there shouldn't be any)
        self.word_dictionary = {}  # dictionary {word: embedding_index}
        for i_word, word in enumerate(known_words):
            self.word_dictionary[word] = i_word + 1  # Index 0 is the padding/unknown index

        self.word_embedding = nn.Embedding(
            len(self.word_dictionary) + 1, self.embedding_dim, padding_idx=self.padding_idx
        )
        self.word_embedding.weight.requires_grad_(
            False
        )  # Training the embedding or not made only a small difference in the past.

        self.lstm = nn.LSTM(self.embedding_dim, self.embedding_dim)

        # Add ReLU and Linear layer: much stronger ✓
        self.linear = nn.Linear(
            self.embedding_dim, 6
        )  # 6-D output: 2D position, 4D quaternion rotation

    def forward(self, captions):
        word_indices = [
            [self.word_dictionary.get(word, self.padding_idx) for word in caption.split()]
            for caption in captions
        ]
        caption_lengths = [len(w) for w in word_indices]
        batch_size, max_length = len(word_indices), max(caption_lengths)
        padded_indices = np.ones((batch_size, max_length), np.int) * self.padding_idx

        for i, caption_length in enumerate(caption_lengths):
            padded_indices[i, :caption_length] = word_indices[i]

        padded_indices = torch.from_numpy(padded_indices)
        if self.is_cuda():
            padded_indices = padded_indices.cuda()

        # Packing seems to work, sentences have the same output regardless of padding
        embedded_words = self.word_embedding(padded_indices)
        x = nn.utils.rnn.pack_padded_sequence(
            embedded_words, torch.tensor(caption_lengths), batch_first=True, enforce_sorted=False
        )

        h = torch.zeros(1, batch_size, self.word_embedding.embedding_dim)
        c = torch.zeros(1, batch_size, self.word_embedding.embedding_dim)
        if self.is_cuda():
            h, c = h.cuda(), c.cuda()

        # Encode description
        _, (h, c) = self.lstm(x, (h, c))

        # Regress 6D output from last hidden state
        h = torch.squeeze(h)
        h = F.relu(h)
        h = self.linear(h)

        # Normalize quaternion
        F.normalize(h[:, 2:6])
        return h

    def is_cuda(self):
        return next(self.parameters()).is_cuda
