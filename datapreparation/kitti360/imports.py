from typing import List
import numpy as np

class Object3d:
    def __init__(self, xyz, rgb, label, id):
        self.xyz = xyz
        self.rgb = rgb
        self.label = label
        self.id = id
        self.closest_point = None # Set in get_closest_point for cell-object

    def __repr__(self):
        return f'Object3d: {self.label}'

    def apply_downsampling(self, indices):
        self.xyz = self.xyz[indices]
        self.rgb = self.rgb[indices]

    def mask_points(self, mask):
        """Mask xyz and rgb, the id is retained
        """
        assert len(mask)>6 # To prevent bbox input
        return Object3d(self.xyz[mask], self.rgb[mask], self.label, self.id)      

    # def center(self):
    #     return 1/2 * (np.min(self.xyz, axis=0) + np.max(self.xyz, axis=0)) 

    def get_closest_point(self, anchor):
        dists = np.linalg.norm(self.xyz - anchor, axis=1)
        self.closest_point = self.xyz[np.argmin(dists)]
        return self.closest_point

    @classmethod
    def merge(cls, obj1, obj2):
        assert obj1.label==obj2.label and obj1.id==obj2.id
        return Object3d(
            np.vstack((obj1.xyz, obj2.xyz)),
            np.vstack((obj1.rgb, obj2.rgb)),
            obj1.label,
            obj1.id
        )

class Description:
    def __init__(self, object_id, direction, object_label):
        self.object_id = object_id
        self.direction = direction
        self.object_label = object_label

    def __repr__(self):
        return f'Pose is {self.direction} of a {self.object_label}'

class Cell:
    def __init__(self, scene_name, objects: List[Object3d], descriptions: List[Description], pose):
        """
        Args:
            IDs should be unique across entire dataset
            Objects include distractors and mentioned, already cropped and normalized in cell
            Pose as (x,y,z), already normalized in cell
        """
        self.scene_name = scene_name
        self.objects = objects
        self.descriptions = descriptions
        self.pose = pose    

    def __repr__(self):
        return f'Cell: {len(self.objects)} objects, {len(self.descriptions)} descriptions'

    def get_text(self):
        text = ""
        for d in self.descriptions:
            text += str(d) + '. '
        return text