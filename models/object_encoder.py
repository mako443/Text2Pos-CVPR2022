import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from models.modules import get_mlp
from models.pointcloud.pointnet2 import PointNet2

class ObjectEncoder(torch.nn.Module):
    def __init__(self, embed_dim, known_classes, known_colors, args):
        super(ObjectEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.args = args

        # Set idx=0 for padding
        self.known_classes = {c: (i+1) for i,c in enumerate(known_classes)}
        self.known_classes['<unk>'] = 0
        self.class_embedding = nn.Embedding(len(self.known_classes), embed_dim, padding_idx=0)

        self.pos_embedding = get_mlp([3, 64, embed_dim]) #OPTION: pos_embedding layers
        self.color_embedding = get_mlp([3, 64, embed_dim]) #OPTION: color_embedding layers        

        self.pointnet = PointNet2(len(known_classes), len(known_colors), args) # The known classes are all the same now, at least for K360
        self.pointnet.load_state_dict(torch.load(args.pointnet_path))
        # self.pointnet_dim = self.pointnet.lin2.weight.size(0)     

        if args.pointnet_freeze:
            print('CARE: freezing PN')  
            self.pointnet.requires_grad_(False)

        if args.pointnet_features == 0:
            self.mlp_pointnet = get_mlp([self.pointnet.dim0, self.embed_dim])
        elif args.pointnet_features == 1:
            self.mlp_pointnet = get_mlp([self.pointnet.dim1, self.embed_dim])     
        elif args.pointnet_features == 2:
            self.mlp_pointnet = get_mlp([self.pointnet.dim2, self.embed_dim])                        
        self.mlp_merge = get_mlp([len(args.use_features)*embed_dim, embed_dim])        

    def forward(self, objects, object_points):
        """Features are currently normed before merging but not at the end.

        Args:
            objects (List[List[Object3d]]): List of lists of objects
            object_points (List[Batch]): List of PyG-Batches of object-points
        """

        if self.args.pointnet_embed:
            class_indices = []
            for i_batch, objects_sample in enumerate(objects):
                for obj in objects_sample:
                    class_idx = self.known_classes.get(obj.label, 0)
                    class_indices.append(class_idx)
        else:
            if self.args.pointnet_features == 0:
                object_features = [self.pointnet(pyg_batch.to(self.device)).features0 for pyg_batch in object_points] # [B, obj_counts, PN_dim]
            elif self.args.pointnet_features == 1:
                object_features = [self.pointnet(pyg_batch.to(self.device)).features1 for pyg_batch in object_points] # [B, obj_counts, PN_dim]
            elif self.args.pointnet_features == 2:
                object_features = [self.pointnet(pyg_batch.to(self.device)).features2 for pyg_batch in object_points] # [B, obj_counts, PN_dim]

            # object_features = [self.pointnet(pyg_batch.to(self.device)).features for pyg_batch in object_points] # [B, obj_counts, PN_dim]
            object_features = torch.cat(object_features, dim=0) # [total_objects, PN_dim]
            object_features = self.mlp_pointnet(object_features)    

            # For select
            # object_class_indices = [self.pointnet(pyg_batch.to(self.device)).class_pred for pyg_batch in object_points] # [B, obj_counts]
            # object_class_indices = torch.cat(object_class_indices, dim=0)
            # object_class_indices = torch.argmax(object_class_indices, dim=-1)               

        embeddings = []
        if 'class' in self.args.use_features:
            if self.args.pointnet_embed: # Use fixed embedding (ground-truth data!)
                class_embedding = self.class_embedding(torch.tensor(class_indices, dtype=torch.long, device=self.device))
                embeddings.append(F.normalize(class_embedding, dim=-1))
            else:
                embeddings.append(F.normalize(object_features, dim=-1)) # Use features from PointNet
                # For select
                # class_embedding = self.class_embedding(object_class_indices)
                # embeddings.append(F.normalize(class_embedding, dim=-1))                
        if 'color' in self.args.use_features:
            colors = []
            for objects_sample in objects:
                colors.extend([obj.get_color_rgb() for obj in objects_sample])
            color_embedding = self.color_embedding(torch.tensor(colors, dtype=torch.float, device=self.device))
            embeddings.append(F.normalize(color_embedding, dim=-1))
        if 'position' in self.args.use_features:
            positions = []
            for objects_sample in objects:
                # positions.extend([obj.center_in_cell for obj in objects_sample])
                positions.extend([obj.closest_point for obj in objects_sample])
            pos_embedding = self.pos_embedding(torch.tensor(positions, dtype=torch.float, device=self.device))
            embeddings.append(F.normalize(pos_embedding, dim=-1)) 

        if len(embeddings) > 1:
            embeddings = self.mlp_merge(torch.cat(embeddings, dim=-1))
        else:
            embeddings = embeddings[0]  

        return embeddings

    @property
    def device(self):
        return next(self.pos_embedding.parameters()).device   

    def get_device(self):
        return next(self.pos_embedding.parameters()).device            