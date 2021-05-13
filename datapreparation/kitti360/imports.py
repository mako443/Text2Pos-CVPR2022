from typing import List
import numpy as np
import cv2

from datapreparation.kitti360.utils import COLORS, COLOR_NAMES

class Object3d:
    # NOTE: use cell-id and scene-names to unique identify objects if needed
    def __init__(self, id, instance_id, xyz, rgb, label):
        self.id = id # Object ID, unique only inside a single cell. Multiple ids can belong to the same instance ID.
        self.instance_id = instance_id # Original instance ID, can repeat across cells and in the same cell due to clustering of stuff objects.
        self.xyz = xyz
        self.rgb = rgb
        self.label = label
        # self.closest_point = None # Set in get_closest_point() for cell-object. CARE: may now be "incorrect" since multiple poses can use this object/cells
        # self.center = None # TODO, for SG-Matching: ok to just input center instead of closest-point? or better to directly input xyz (care that PN++ knowns about coords)

    def get_color_rgb(self):
        color = np.mean(self.rgb, axis=0)
        assert color.shape == (3,)
        return color

    def get_color_text(self):
        """Get the color as text based on the closest (L2) discrete color-center.
        CARE: Can change during downsampling or masking
        """
        dists = np.linalg.norm(np.mean(self.rgb, axis=0) - COLORS, axis=1)
        return COLOR_NAMES[np.argmin(dists)]

    def get_center(self):
        return np.mean(self.xyz, axis=0)

    def __repr__(self):
        return f'Object3d: {self.label}'

    def apply_downsampling(self, indices):
        self.xyz = self.xyz[indices]
        self.rgb = self.rgb[indices]

    def mask_points(self, mask):
        """Mask xyz and rgb, the id is retained
        """
        assert len(mask)>6 # To prevent bbox input
        # return Object3d(self.xyz[mask], self.rgb[mask], self.label, self.id)    
        return Object3d(self.id, self.instance_id, self.xyz[mask], self.rgb[mask], self.label)  

    # def center(self):
    #     return 1/2 * (np.min(self.xyz, axis=0) + np.max(self.xyz, axis=0)) 

    def get_closest_point(self, anchor):
        dists = np.linalg.norm(self.xyz - anchor, axis=1)
        # self.closest_point = self.xyz[np.argmin(dists)]
        return self.xyz[np.argmin(dists)]

    @classmethod
    def merge(cls, obj1, obj2):
        assert obj1.label==obj2.label and obj1.id==obj2.id, f'{obj1.label}, {obj2.label}, {obj1.id}, {obj2.id}'
        return Object3d(
            obj1.id, obj1.instance_id,
            np.vstack((obj1.xyz, obj2.xyz)),
            np.vstack((obj1.rgb, obj2.rgb)),
            obj1.label
        )
        # return Object3d(
        #     np.vstack((obj1.xyz, obj2.xyz)),
        #     np.vstack((obj1.rgb, obj2.rgb)),
        #     obj1.label,
        #     obj1.id
        # )

    @classmethod
    def create_padding(cls):
        # obj = Object3d(np.random.rand(8,3) * 0.001, np.zeros((8,3)), 'pad', -1) # Creating too few points or zero positios throws nans in PyG
        obj = Object3d(-1, -1, np.random.rand(8,3) * 0.001, np.zeros((8,3)), 'pad')
        obj.get_closest_point([-1, -1, -1])
        return obj

class Description:
    def __init__(self, object_id, object_instance_id, direction, object_label, object_color, object_closest_point):
        self.object_id = object_id
        self.object_instance_id = object_instance_id
        self.direction = direction
        self.object_label = object_label
        self.object_color = object_color
        self.object_closest_point = object_closest_point
        assert (object_color is not None)

    def __repr__(self):
        return f'Pose is {self.direction} of a {self.object_color} {self.object_label}'

class Pose:
    def __init__(self, pose_in_cell, pose_w, best_cell_id, descriptions: List[Description]):
        self.pose = pose_in_cell # The pose in the best cell (specified by best_cell_id), normed to ∈ [0, 1]
        self.pose_w = pose_w
        self.best_cell_id = best_cell_id
        self.descriptions = descriptions

    def __repr__(self) -> str:
        return f'Pose at {self.pose_w} in {self.cell_id}'

    def get_text(self):
        text = ""
        for d in self.descriptions:
            text += str(d) + '. '
        return text    

class Cell:
    def __init__(self, idx, scene_name, objects: List[Object3d], cell_size, bbox_w):
        """
        Args:
            IDs should be unique across entire dataset
            Objects include distractors and mentioned, already cropped and normalized in cell
            Pose as (x,y,z), already normalized in cell
            cell-size: longest edge in world-coordinates
            Pose_w as (x,y,z) in original world-coordinates
        """
        self.id = f'{scene_name}_{idx:04.0f}' # Incrementing alpha-numeric id in format 00XX_XXXX
        assert len(self.id) == 9, self.id
        self.objects = objects
        # self.descriptions = descriptions
        # self.pose = pose    
        
        self.cell_size = cell_size # Original cell-size (longest edge)
        # self.pose_w = pose_w # Original pose in world-coordinates
        self.bbox_w = bbox_w # Original pose in world-coordinates

    def __repr__(self):
        return f'Cell {self.id}: {len(self.objects)} objects at {np.int0(self.bbox_w)}'

    # def get_text(self):
    #     text = ""
    #     for d in self.descriptions:
    #         text += str(d) + '. '
    #     return text