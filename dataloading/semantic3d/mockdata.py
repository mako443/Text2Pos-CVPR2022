import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import os.path as osp
import pickle

class MockDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,index):
        image  =  Image.open(self.image_paths[index]).convert('RGB')     
        if self.transform:
            image=self.transform(image)

        return image    