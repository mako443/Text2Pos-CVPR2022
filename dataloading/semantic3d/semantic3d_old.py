import numpy as np
import cv2
import pickle
import os
import os.path as osp
from torch.utils.data import Dataset

from datapreparation.imports import Object3D, Pose, DescriptionObject
from datapreparation.drawing import draw_objects_poses, draw_objects_poses_descriptions

# import sys
# sys.path.append('/home/imanox/Documents/Text2Image/Semantic3D-Net')
# import semantic.imports
# from graphics.imports import CLASSES_COLORS, Pose, CORNERS
# from drawing.utils import plot_objects

'''
TODO:
- pad/choice for equal lengths
- Norm positions for multiple scenes
'''
class Semantic3dObjectData(Dataset):
    def __init__(self, root, scene_name, description_length=12):
        self.root = root
        self.scene_name= scene_name
        self.description_length = 8
        
        self.objects = pickle.load(open(osp.join(root, 'train', scene_name, 'objects.pkl'), 'rb'))
        self.poses = pickle.load(open(osp.join(root, 'train', scene_name, 'poses.pkl'), 'rb'))
        self.pose_descriptions = pickle.load(open(osp.join(root, 'train', scene_name, 'descriptions.pkl'), 'rb'))
        self.keys = sorted(list(self.pose_descriptions.keys()))

        self.object_positions = []
        for obj in self.objects:
            self.object_positions.append(0.5*(np.max(obj.points_w[:,0:2], axis=0) + np.min(obj.points_w[:,0:2], axis=0)))
        self.object_positions = np.array(self.object_positions)
        self.object_classes = [o.label for o in self.objects]
        self.object_ids = [o.id for o in self.objects]

        print(str(self))

    def __str__(self):
        return f'Semantic3dObjectData: {self.scene_name} with {len(self.objects)} objects from, {len(self.poses)} poses'

    def plot_scene(self):
        return draw_objects_poses(self.objects, self.poses, draw_arrows=False), 0

    def plot_pose(self, idx):
        k = self.keys[idx]
        return draw_objects_poses_descriptions(self.objects, (self.poses[k],), (self.pose_descriptions[k], ))

    def __getitem__(self, index):
        key = self.keys[index]     

        description = self.pose_descriptions[key]
        #Select object if too many...
        if len(description)>self.description_length:
            description_objects = list(np.random.choice(description, size=self.description_length, replace=False))
            description_directions = [d.direction for d in description_objects]   
            description_classes = [d.label for d in description_objects]   
            pads = 0
        #... or pad if too few
        elif len(description)<self.description_length:
            pads = self.description_length-len(description)
            description_objects = description
            description_directions = [d.direction for d in description_objects] + ['<pad>' for _ in range(pads)]
            description_classes = [d.label for d in description_objects] + ['<pad>' for _ in range(pads)]

        angle = self.poses[key].phi
        match_indices = [self.object_ids.index(d.id) for d in description_objects]   

        return {'object_positions': self.object_positions,
                'object_classes': self.object_classes,
                'description_directions': description_directions,
                'description_classes': description_classes,
                'angles': angle,
                'match_indices': match_indices,
                'description_lengths': len(description)}

    def __len__(self):
        return 8
        return len(self.keys)

    #TODO: refer to specific names
    def collate_fn(samples):
        collated = {}
        for key in samples[0].keys():
            if type(samples[0][key])==list:
                collated[key] = [sample[key] for sample in samples]
            elif type(samples[0][key]) in (int, np.float64): #Angles and description-lengths
                collated[key] = np.array([sample[key] for sample in samples])
            elif type(samples[0][key])==np.ndarray:
                collated[key] = [sample[key] for sample in samples]
            else:
                raise Exception('Unexpected type in dataset! '+str(type(samples[0][key])))

        return collated        


if __name__ == "__main__":
    dataset = Semantic3dObjectData('./data/semantic3d', 'sg27_station5_intensity_rgb')
    idx = 0
    key = dataset.keys[idx]
    description = dataset.pose_descriptions[key]
    for d in description: print(d)

    img = cv2.flip(dataset.plot_pose(idx), 0)
    cv2.imshow("", img); cv2.waitKey()
    