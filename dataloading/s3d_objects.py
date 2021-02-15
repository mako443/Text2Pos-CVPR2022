import os
import sys
import time
import h5py
import json
import pickle
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset

from datapreparation.imports import Object3D, DescriptionObject

class Semantic3dReferenceDataset(Dataset):
    def __init__(self, path_numpy, path_scenes):
        self.path_numpy = path_numpy
        self.path_scenes = path_scenes
    
        self.scene_name = 'sg27_station5_intensity_rgb'

        #Load objects
        self.scene_objects     = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'objects.pkl'), 'rb'))
        self.list_descriptions = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'list_object_descriptions.pkl'), 'rb'))
        self.text_descriptions = pickle.load(open(osp.join(self.path_scenes,'train',self.scene_name,'text_object_descriptions.pkl'), 'rb'))
        

