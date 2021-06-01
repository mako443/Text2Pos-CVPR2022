from typing import List

import numpy as np
import random
from datapreparation.kitti360.imports import Object3d, get_R

def get_direction(obj, pose):
    raise Exception('Not usable with orientation.')
    closest_point = obj.get_closest_point(pose)
    obj2pose = pose - closest_point
    if np.linalg.norm(obj2pose[0:2]) < 0.05: # Before: 0.015
        direction = 'on-top'
    else:
        if abs(obj2pose[0])>=abs(obj2pose[1]) and obj2pose[0]>=0: direction='east'
        if abs(obj2pose[0])>=abs(obj2pose[1]) and obj2pose[0]<=0: direction='west'
        if abs(obj2pose[0])<=abs(obj2pose[1]) and obj2pose[1]>=0: direction='north'
        if abs(obj2pose[0])<=abs(obj2pose[1]) and obj2pose[1]<=0: direction='south'  
    return direction

def get_direction_phi(obj: Object3d, pose: np.ndarray, phi: float):
    """
    Does not use on-top, all based on center. Better idea?
    NOTE: Direction now flipped to "there is a <color> <class> <direction> of the pose.
    """
    pose2obj = obj.get_center() - pose
    if abs(pose2obj[0])>=abs(pose2obj[1]) and pose2obj[0]>=0: direction='right'
    if abs(pose2obj[0])>=abs(pose2obj[1]) and pose2obj[0]<=0: direction='left'
    if abs(pose2obj[0])<=abs(pose2obj[1]) and pose2obj[1]>=0: direction='ahead'
    if abs(pose2obj[0])<=abs(pose2obj[1]) and pose2obj[1]<=0: direction='behind'

    R = get_R(phi)
    pose2obj_phi = R @ pose2obj[0:2] 
    if abs(pose2obj_phi[0])>=abs(pose2obj_phi[1]) and pose2obj_phi[0]>=0: direction_phi='right'
    if abs(pose2obj_phi[0])>=abs(pose2obj_phi[1]) and pose2obj_phi[0]<=0: direction_phi='left'
    if abs(pose2obj_phi[0])<=abs(pose2obj_phi[1]) and pose2obj_phi[1]>=0: direction_phi='ahead'
    if abs(pose2obj_phi[0])<=abs(pose2obj_phi[1]) and pose2obj_phi[1]<=0: direction_phi='behind'    

    return direction, direction_phi

def select_objects_closest(objects: List[Object3d], pose, num_mentioned):
    distances = np.linalg.norm([obj.get_closest_point(pose) - pose for obj in objects], axis=1)
    indices = np.argsort(distances)[0 : num_mentioned]
    return [objects[i] for i in indices]

def select_objects_direction(objects: List[Object3d], pose, num_mentioned, phi):
    #directions = [get_direction(obj, pose) for obj in objects]
    directions = [get_direction_phi(obj, pose, phi)[1] for obj in objects]
    direction_indices = {d: [] for d in directions}
    for idx, d in enumerate(directions):
        direction_indices[d].append(idx)

    keys = list(direction_indices.keys())
    # random.shuffle(keys)

    offset = 0
    indices = []
    while len(indices) < num_mentioned:
        for key in keys:
            value = direction_indices[key]
            if len(value) > offset:
                indices.append(value[offset])
        offset += 1
    indices = indices[0 : num_mentioned]
    return [objects[i] for i in indices]

def select_objects_class(objects: List[Object3d], pose, num_mentioned):
    class_indices = {obj.label: [] for obj in objects}
    for idx, obj in enumerate(objects):
        class_indices[obj.label].append(idx)

    keys = list(class_indices.keys())
    # random.shuffle(keys)

    offset = 0
    indices = []
    while len(indices) < num_mentioned:
        for key in keys:
            value = class_indices[key]
            if len(value) > offset:
                indices.append(value[offset])
        offset += 1
    indices = indices[0 : num_mentioned]
    return [objects[i] for i in indices]

def select_objects_random(objects: List[Object3d], pose, num_mentioned):
    return list(np.random.choice(objects, size=num_mentioned, replace=False))
