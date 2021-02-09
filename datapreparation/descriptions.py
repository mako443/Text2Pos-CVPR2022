import numpy as np
import os
import os.path as osp
import pickle
import cv2

from datapreparation.imports import Object3D, ViewObject, Pose, DescriptionObject, DIRECTIONS, DIRECTIONS_COMPASS, IMAGE_WIDHT, calc_angle_diff
from datapreparation.drawing import draw_objects_poses, draw_objects_poses_viewObjects

'''
Module to describe the Semantic3D poses based on their view-objects
- TODO/CARE: Can select same object for multiple directions
'''

'''
Descriptions for poses
'''

#Retain view-objects in middle half of image, then sort based on the clostest distances to the image-center, take up to <max_objects>
def select_view_objects(view_objects, pose, max_objects):
    view_objects = [vo for vo in view_objects if np.abs(vo.center[0] - 1/2*IMAGE_WIDHT) <= 1/2*IMAGE_WIDHT]
    view_objects_sorted = sorted(view_objects, key= lambda vo: np.abs(vo.center[0] - 1/2*IMAGE_WIDHT)) #CARE: assumes [0] is always x-direction
    return view_objects_sorted[0:max_objects]

#Describes the pose at <key> based on up to <max_objects> objects in each direction, chosing the most-center ones
def describe_pose(view_objects, poses, key, max_objects=3):
    ref = poses[key]

    #Search the key of every pose that will be used for each direction
    direction_keys = {}
    for direction, angle in DIRECTIONS.items():
        min_ad = np.inf
        for key, pose in poses.items():
            angle_diff = calc_angle_diff(ref.phi+angle, pose.phi)
            min_ad = np.minimum(min_ad, angle_diff)
            if np.linalg.norm(ref.eye-pose.eye)<0.5 and angle_diff<np.pi/8:
                direction_keys[direction] = key
                break

    print(direction_keys)
    assert len(direction_keys)==4

    description = []
    for direction, key in direction_keys.items():
        selected_view_objects = select_view_objects(view_objects[key], poses[key], max_objects)
        description.extend([DescriptionObject.from_view_object(vo, direction) for vo in selected_view_objects])

    return description

def get_text_description(description):
    text = ""
    for direction in DIRECTIONS.keys():
        direction_objects = [o for o in description if o.direction==direction]
        if direction_objects:
            text += (f'{direction} I see ')
            for i_do, do in enumerate(direction_objects):
                text += (f'a {do.color_text} {do.label}')
                if i_do<len(direction_objects)-1: 
                    text += (' and ')
                else: 
                    text += ('. ')
    return text

'''
Descriptions for objects
'''

'''
Describes an object based on the closest ones next to it
CARE: the objects are not necessarily visible from e/o positions
Directions: +y <=> north, +x <=> west
'''
def describe_object(scene_objects, idx, max_mentioned_objects=5, max_dist=25):
    description = []
    
    #Add the target
    target = scene_objects[idx]
    description.append(DescriptionObject.from_object3d(target, None)) # Direction=None means the object is the target

    #Add the closest other objects
    object_distances = []
    for obj in scene_objects:
        if obj.id==target.id:
            object_distances.append(np.inf*np.ones(2))
        else:
            object_distances.append(target.center - obj.center)
    object_distances = np.linalg.norm(object_distances, axis=1)
    
    closest_indices = np.argsort(object_distances)[0:max_mentioned_objects]
    for closest_index in closest_indices:
        diff = target.center - scene_objects[closest_index].center
        if abs(diff[0])>=abs(diff[1]) and diff[0]>=0: direction='east'
        if abs(diff[0])>=abs(diff[1]) and diff[0]<=0: direction='west'
        if abs(diff[0])<=abs(diff[1]) and diff[1]>=0: direction='south'
        if abs(diff[0])<=abs(diff[1]) and diff[1]<=0: direction='north'

        description.append(DescriptionObject.from_object3d(scene_objects[closest_index], direction))

    text = ""
    target = description[0]
    assert target.direction is None
    text += f'It is a {target.color_text} {target.label}'
    for i, obj in enumerate(description):
        if i==0: continue
        if i==1:
            text += f' that is {obj.direction} of a {obj.color_text} {obj.label}'
        if i>1:
            text += f' and {obj.direction} of a {obj.color_text} {obj.label}'
    text += '.'

    return description, text


#Graph with objects and attributes as nodes, pairwise bidrectionals connections between all objects to facilitate learning
def get_graph_description():
    pass