import numpy as np
import cv2
import pickle
import os
import os.path as osp
from torch.utils.data import Dataset
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R

from datapreparation.imports import Object3D, ViewObject, Pose, DescriptionObject, calc_angle_diff

class TextDescriptionData(Dataset):
    def __init__(self, root, split, scene_name, split_mask=None):
        self.root = root
        self.split = split
        self.scene_name = scene_name

        self.poses = pickle.load(open(osp.join(root, split, scene_name, 'poses.pkl'), 'rb'))
        self.poses_texts = pickle.load(open(osp.join(root, split, scene_name, 'description_texts.pkl'), 'rb'))
        assert len(self.poses)==len(self.poses_texts)
        
        self.poses_keys = np.array(sorted(self.poses.keys()))

        if split_mask is not None:
            assert len(split_mask)==len(self.poses_keys)
            self.poses_keys = self.poses_keys[split_mask]

        #Pre-process the texts
        for key in self.poses_keys:
            self.poses_texts[key] = self.trim_text(self.poses_texts[key])
        
        self.poses_6d = np.zeros((len(self), 6), dtype=np.float32)
        for i_key, key in enumerate(self.poses_keys):
            self.poses_6d[i_key, 0:2] = self.poses[key].eye[0:2]
            self.poses_6d[i_key, 2:6] = R.from_rotvec((0, 0, self.poses[key].phi)).as_quat()

        print(str(self))

    def __len__(self):
        return len(self.poses_keys)

    def __str__(self):
        return f'TextDescriptionData: {self.root} - {self.split} - {self.scene_name}, {len(self)} poses'

    def __getitem__(self, index):
        k = self.poses_keys[index]
        text = self.poses_texts[k] 
        return {'descriptions': self.poses_texts[k], 'poses': self.poses_6d[index]}

    def trim_text(self, text):
        return text.strip().lower().replace('.', '').replace(',', '')

    def get_known_words(self):
        words = []
        for text in self.poses_texts.values():
            words.extend(text.split())
        return list(np.unique(words))


if __name__ == "__main__":
    data = TextDescriptionData('data/semantic3d', 'train', 'sg27_station5_intensity_rgb')


    
    
    