import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import os
import pickle
from easydict import EasyDict

from models.modules import get_mlp, LanguageEncoder

"""
Transformer-based matching modules

TODO:
- why is "wrong" order better??
- encode obj color
- encode position or full bboxes?
- use InstanceNorm / LayerNorm

PARAMETERS:
- pos_embedding
- language_encoder
- nhead, dim_feedforward
- MLPs
"""


class TransformerMatch1(torch.nn.Module):
    def __init__(
        self, known_classes, known_words, embedding_dim, num_layers, nhead, dim_feedforward
    ):
        super(TransformerMatch1, self).__init__()

        # Set idx=0 for padding
        self.known_classes = {c: (i + 1) for i, c in enumerate(known_classes)}
        self.known_classes["<unk>"] = 0
        self.class_embedding = nn.Embedding(len(self.known_classes), embedding_dim, padding_idx=0)

        # TODO: optimize these layers
        # self.pos_embedding = get_mlp([2,4,8,16,embedding_dim])
        self.pos_embedding = get_mlp([2, 128, embedding_dim])

        self.language_encoder = LanguageEncoder(known_words, embedding_dim, bi_dir=True)

        # Self attention layers
        # self.encoder_layers = nn.ModuleList([nn.TransformerEncoderLayer(2 * embedding_dim, nhead=8, dim_feedforward=2048) for i in range(num_layers)])
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    2 * embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward
                )
                for i in range(num_layers)
            ]
        )
        # TODO: do Xavier?
        for layer in self.encoder_layers:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        # MLPs
        self.mlp_ref_confidence = get_mlp([2 * embedding_dim, embedding_dim, 1])

        self.mlp_target_class = get_mlp(
            [embedding_dim, embedding_dim // 2, len(self.known_classes)]
        )

        self.mlp_object_class = get_mlp([2 * embedding_dim, embedding_dim, len(self.known_classes)])

        self.mlp_mentioned = get_mlp([2 * embedding_dim, embedding_dim, 1])

        # self.mlp_object_offset = get_mlp([2*embedding_dim, embedding_dim, 64, 2])

    # TODO: if objects are the same, feed them in once or as full batch? -> Probably as batch for separate aux.-losses
    """
    Objects as [obj1, obj2, ...], assumed to be the same for all descriptions
    Descriptions as batch [d1, d2, d3, ..., d_B] with d_i a string. Strings can be of different sizes.
    ## Object referance types [B, num_obj ∈ {0,1,2}: ground-truth whether each object is unrelated, mentioned, or the target. Used for aux.-loss.
    """

    def forward(self, object_classes, object_positions, descriptions, do_print=False):
        """
        Encode the descriptions
        """
        batch_size = len(descriptions)
        description_encodings = self.language_encoder(descriptions)  # [B, DIM]
        description_encodings = torch.unsqueeze(description_encodings, dim=1)  # [B, 1, DIM]

        # description_encodings = F.relu(description_encodings) #TODO: do this or not?

        """
        Encode the objects
        """
        num_objects = len(object_classes[0])
        class_indices = torch.zeros((batch_size, num_objects), dtype=torch.long)
        for i in range(batch_size):
            for j in range(num_objects):
                class_indices[i, j] = self.known_classes.get(object_classes[i][j], 0)
        class_embeddings = self.class_embedding(class_indices.to(self.device))  # [B, num_obj, DIM]

        pos_embeddings = self.pos_embedding(
            torch.tensor(object_positions, dtype=torch.float, device=self.device)
        )  # [B, num_obj, DIM]

        object_encodings = class_embeddings + pos_embeddings  # [B, num_obj, DIM]
        if do_print:
            print()
            print(
                "class",
                torch.mean(torch.abs(class_embeddings)).item(),
                "pos",
                torch.mean(torch.abs(pos_embeddings)).item(),
            )
            print()

        # TODO: norm somewhere?
        """
        Merge object and description encodings (concat the description encoding to every object encoding for combined transformer inputs)
        """
        description_encodings_repeated = description_encodings.repeat(
            1, num_objects, 1
        )  # [B, num_obj, DIM]
        transformer_input = torch.cat(
            (object_encodings, description_encodings_repeated), dim=-1
        )  # [B, num_obj, 2*DIM]

        """
        Run Tranformer Encoder Layers
        """
        # features = torch.transpose(transformer_input, 0, 1) # [B, num_obj, E] -> [num_obj, B, E]
        features = transformer_input  # TODO: transpose or not?!
        for layer in self.encoder_layers:
            features = layer(features)

        """
        Make predictions
        """
        obj_ref_pred = self.mlp_ref_confidence(features)  # [num_obj, B, 1]
        # obj_ref_pred = torch.transpose(torch.squeeze(obj_ref_pred, dim=-1), 0, 1) # [num_obj, B, 1] -> [B, num_obj]
        obj_ref_pred = torch.squeeze(obj_ref_pred, dim=-1)

        target_class_pred = torch.squeeze(self.mlp_target_class(description_encodings), dim=1)

        obj_class_pred = self.mlp_object_class(features)

        obj_mentioned_pred = self.mlp_mentioned(features)

        # obj_offset_pred = self.mlp_object_offset(features)

        model_output = EasyDict()
        model_output.features = features
        model_output.obj_ref_pred = obj_ref_pred
        model_output.target_class_pred = target_class_pred
        model_output.obj_class_pred = obj_class_pred
        model_output.obj_mentioned_pred = obj_mentioned_pred
        # model_output.obj_offset_pred = obj_offset_pred

        return model_output

    @property
    def device(self):
        return next(self.pos_embedding.parameters()).device


if __name__ == "__main__":
    model = TransformerMatch1(
        ["high vegetation", "low vegetation", "buildings", "hard scape", "cars"],
        "a b c d e".split(),
        300,
        2,
    )

    model(
        [
            ["high vegetation", "low vegetation", "buildings", "hard scape", "cars", "xx"]
            for i in range(3)
        ],
        ["a b x d e", "a b c x a c", "a a"],
    )
