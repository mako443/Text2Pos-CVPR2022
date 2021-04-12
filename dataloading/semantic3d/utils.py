import os
import os.path as osp
import pickle
import numpy as np
import torch_geometric
import torch

CLASSES= ['class0', 'class1','class2']
ATTRIBUTES= ['left-of', 'right-of', 'above-of', 'below-of']

def create_embedding_dictionaries():
    vertex_dict= {}
    for i,c in enumerate(CLASSES):
        vertex_dict[c]= i

    edge_dict= {}
    for i,a in enumerate(ATTRIBUTES):
        edge_dict[a]= i

    return vertex_dict, edge_dict
