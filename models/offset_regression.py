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

        self.language_encoder = LanguageEncoder(known_words, args.embed_dim, bi_dir=True)
        self.mlp_offsets = get_mlp_offset([args.embed_dim, args.embed_dim // 2, 2])

