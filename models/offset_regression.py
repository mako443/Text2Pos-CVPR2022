from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.modules import LanguageEncoder
from models.superglue_matcher import get_mlp_offset

class OffsetRegressor(torch.nn.Module):
    def __init__(self, known_words, args):
        super(OffsetRegressor, self).__init__()
        self.args = args

        self.language_encoder = LanguageEncoder(known_words, args.regressor_dim, bi_dir=True)
        self.mlp_offsets = get_mlp_offset([args.regressor_dim, args.regressor_dim // 2, 2])

    def forward(self, hints):
        hint_encodings = torch.stack([self.language_encoder(hint_sample) for hint_sample in hints]) # [B, num_hints, DIM]
        # hint_encodings = F.normalize(hint_encodings, dim=-1) #Norming those too

        offsets = self.mlp_offsets(hint_encodings) # [B, num_hints, 2]
        return offsets