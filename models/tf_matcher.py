"""Module to perform hints-to-objects matching through a Transformer followed by Sinkhorn iterations.
Unlike SuperGlue, this did not show promising results, only kept for the sake of completion.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import time
import numpy as np
import os
import pickle
from easydict import EasyDict
from scipy.spatial.distance import cdist

from models.modules import get_mlp, LanguageEncoder
from models.superglue import SuperGlue

from dataloading.semantic3d.semantic3d import Semantic3dPoseReferenceMockDataset

"""
Explicit hint <-> object matching w/ Transformer and Sinkhorn
"""


class TransformerMatch(torch.nn.Module):
    def __init__(self, known_classes, known_words, args):
        super(TransformerMatch, self).__init__()
        embed_dim = args.embed_dim
        use_features = args.use_features
        self.embed_dim = args.embed_dim
        self.use_features = args.use_features

        """
        Object path
        """
        # Set idx=0 for padding
        self.known_classes = {c: (i + 1) for i, c in enumerate(known_classes)}
        self.known_classes["<unk>"] = 0
        self.class_embedding = nn.Embedding(len(self.known_classes), embed_dim, padding_idx=0)

        self.pos_embedding = get_mlp([3, 128, embed_dim])  # OPTION: pos_embedding layers
        self.color_embedding = get_mlp([3, 128, embed_dim])  # OPTION: color_embedding layers

        self.mlp_merge = get_mlp([len(use_features) * embed_dim, embed_dim])

        """
        Textual path
        """
        self.language_encoder = LanguageEncoder(known_words, embed_dim, bi_dir=True)

        """
        Transformer and matcher (SuperGlue w/ empty GNN, i.e. just the Sinkhorn layer)
        """
        self.transformer_obj = nn.TransformerEncoderLayer(args.embed_dim, args.nhead, args.dim_ff)
        self.transformer_txt = nn.TransformerEncoderLayer(args.embed_dim, args.nhead, args.dim_ff)
        config = {
            "descriptor_dim": args.embed_dim,
            "GNN_layers": [],
            "sinkhorn_iterations": args.sinkhorn_iters,
            "match_threshold": 0.2,
        }
        self.superglue = SuperGlue(config)

        print(
            f"TransformerMatch: dim: {self.embed_dim}, features: {use_features}, nhead: {args.nhead}, dim_ff: {args.dim_ff}, iters: {args.sinkhorn_iters}"
        )

    def forward(self, objects, hints, verbose=False):
        assert len(objects) == len(hints)
        batch_size = len(objects)
        num_objects = len(objects[0])
        num_hints = len(hints[0])

        """
        Encode the hints
        """
        hint_encodings = torch.stack(
            [self.language_encoder(hint_sample) for hint_sample in hints]
        )  # [B, num_hints, DIM]

        """
        Encode the objects
        """
        embeddings = []
        if "class" in self.use_features:
            class_indices = torch.zeros(
                (batch_size, num_objects), dtype=torch.long, device=self.device
            )
            for i in range(batch_size):
                for j in range(num_objects):
                    class_indices[i, j] = self.known_classes.get(objects[i][j].label, 0)
            class_embedding = self.class_embedding(class_indices)
            embeddings.append(F.normalize(class_embedding, dim=-1))
        if "color" in self.use_features:
            colors = []
            for objects_sample in objects:
                colors.append([obj.color for obj in objects_sample])
            color_embedding = self.color_embedding(
                torch.tensor(colors, dtype=torch.float, device=self.device)
            )
            embeddings.append(F.normalize(color_embedding, dim=-1))
        if "position" in self.use_features:
            positions = []
            for objects_sample in objects:
                positions.append([obj.center for obj in objects_sample])
            pos_embedding = self.pos_embedding(
                torch.tensor(positions, dtype=torch.float, device=self.device)
            )
            embeddings.append(F.normalize(pos_embedding, dim=-1))

        if len(embeddings) > 1:
            obj_encodings = self.mlp_merge(torch.cat(embeddings, dim=-1))
        else:
            obj_encodings = embeddings[0]
        # [B, num_obj, DIM]

        """
        Run Transformer and Matching
        """
        if verbose:
            print("before")
            d = cdist(
                obj_encodings[0].cpu().detach().numpy(), hint_encodings[0].cpu().detach().numpy()
            )
            print(d.astype(np.float16))
        obj_encodings = self.transformer_obj(obj_encodings)  # [B, num_obj, DIM]
        hint_encodings = self.transformer_txt(hint_encodings)  # [B, num_hint, DIM]
        if verbose:
            print("after")
            d = cdist(
                obj_encodings[0].cpu().detach().numpy(), hint_encodings[0].cpu().detach().numpy()
            )
            print(d.astype(np.float16))

        obj_encodings = F.normalize(obj_encodings, dim=-1)
        hint_encodings = F.normalize(hint_encodings, dim=-1)

        desc0 = obj_encodings.transpose(1, 2)  # [B, DIM, num_obj]
        desc1 = hint_encodings.transpose(1, 2)  # [B, DIM, num_hints]

        matcher_output = self.superglue(desc0, desc1)

        outputs = EasyDict()
        outputs.P = matcher_output["P"]
        outputs.matches0 = matcher_output["matches0"]
        outputs.matches1 = matcher_output["matches1"]

        return outputs

    @property
    def device(self):
        return next(self.pos_embedding.parameters()).device


if __name__ == "__main__":
    args = EasyDict()
    args.embed_dim = 16
    args.nhead = 4
    args.dim_ff = 1024
    args.use_features = ["class", "position", "color"]
    args.sinkhorn_iters = 10

    model = TransformerMatch(
        ["high vegetation", "low vegetation", "buildings", "hard scape", "cars"],
        "a b c d e".split(),
        args,
    )

    dataset = Semantic3dObjectReferanceDataset(
        "./data/numpy_merged/", "./data/semantic3d", num_distractors=2
    )
    dataloader = DataLoader(
        dataset, batch_size=2, collate_fn=Semantic3dObjectReferanceDataset.collate_fn
    )
    data = dataset[0]
    batch = next(iter(dataloader))

    out = model.forward(batch["objects"], batch["hint_descriptions"])
