import numpy as np
import os
import os.path as osp
import pickle
import cv2

from typing import List

from datapreparation.semantic3d.imports import Object3D, ViewObject, Pose, DescriptionObject, DIRECTIONS, DIRECTIONS_COMPASS, IMAGE_WIDHT, calc_angle_diff, Cell, CellObject, COLORS, COLOR_NAMES
from datapreparation.semantic3d.drawing import draw_objects_poses, draw_objects_poses_viewObjects

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
def describe_pose_DEPRECATED(view_objects, poses, key, max_objects=3):
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
CARE: max-dist not used!
Directions: +y <=> north, +x <=> east ✓ pptk-checked
'''
def describe_object(scene_objects, idx, max_mentioned_objects=5, max_dist=25):
    description = []
    
    #Add the target
    target = scene_objects[idx]
    description.append(DescriptionObject.from_object3d(target, None)) # Direction=None means the object is the target

    #Add the closest other objects (currently only 2D)
    object_distances = []
    for obj in scene_objects:
        if obj.id==target.id:
            object_distances.append(np.inf*np.ones(2))
        else:
            object_distances.append(target.center[0:2] - obj.center[0:2])
    object_distances = np.linalg.norm(object_distances, axis=1)
    
    closest_indices = np.argsort(object_distances)[0:max_mentioned_objects]
    for closest_index in closest_indices:
        diff = target.center - scene_objects[closest_index].center
        if abs(diff[0])>=abs(diff[1]) and diff[0]>=0: direction='east'
        if abs(diff[0])>=abs(diff[1]) and diff[0]<=0: direction='west'
        if abs(diff[0])<=abs(diff[1]) and diff[1]>=0: direction='north'
        if abs(diff[0])<=abs(diff[1]) and diff[1]<=0: direction='south'

        description.append(DescriptionObject.from_object3d(scene_objects[closest_index], direction))

    # Describe the target
    text = ""
    target = description[0]
    assert target.direction is None
    text += f'It is a {target.color_text} {target.label}'

    # Describe its surroundings
    for i, obj in enumerate(description):
        if i==0: continue
        if i==1:
            text += f' that is {obj.direction} of a {obj.color_text} {obj.label}'
        if i>1:
            text += f' and {obj.direction} of a {obj.color_text} {obj.label}'
    text += '.'

    # Create hints (the same as 'text' but split into a list of separate strings).
    hints = []
    hints.append(f'The target is a {description[0].color_text} {description[0].label}.')
    for obj in description[1:]:
        hints.append(f'It is {obj.direction} of a {obj.color_text} {obj.label}.')

    return description, text, hints

#TODO: use mask to get number of points in cell but still save *all* points (shifted by mean), see Semantic3dPoseReferanceDataset
def describe_cell(scene_objects, cell_bbox, min_fraction=0.33, min_objects=2):
    """Describe a cell by finding the objects inside it

    Args:
        scene_objects (List[Object3D]): List of objects in the scene
        cell_bbox (np.ndarray): [x0, y0, x1, y1]
        min_fraction (float, optional): Minimum fraction of points of an object that have to be inside the cell to count it as one of the cell object. Defaults to 0.33.
        min_objects (int, optional): Minimum number of cell objects, otherwise returns None. Defaults to 2.

    Returns:
        cell (Cell) or None: Cell with data and cell-objects
    """
    assert scene_objects[0].scene_name == scene_objects[-1].scene_name

    cell_size = cell_bbox[2] - cell_bbox[0]
    DIRECTIONS = np.array([ [0, 1],
                            [0,-1],
                            [ 1, 0],
                            [-1, 0],
                            [ 0, 0],]) * cell_size/3
    
    DIRECTION_NAMES = ['north', 'south', 'east', 'west', 'center']

    cell_mean = np.hstack((0.5*(cell_bbox[0:2] + cell_bbox[2:4]), 0))
    cell_objects = []
    for obj in scene_objects:
        mask = np.bitwise_and.reduce((  obj.points_w[:, 0] >= cell_bbox[0], 
                                        obj.points_w[:, 1] >= cell_bbox[1],
                                        obj.points_w[:, 0] <= cell_bbox[2],
                                        obj.points_w[:, 1] <= cell_bbox[3]))
        points_in_cell = obj.points_w[mask]
        points_in_cell_color = obj.points_w_color[mask]
        if len(points_in_cell) / len(obj.points_w) >= min_fraction:
            points_in_cell_relative = points_in_cell - cell_mean
            cell_object = CellObject(obj.points_w, obj.points_w_color, points_in_cell_relative, points_in_cell_color, obj.label, obj.id, obj.color, obj.scene_name)
            cell_objects.append(cell_object)

    if len(cell_objects) < min_objects:
        return None

    cell = Cell(cell_bbox, scene_objects[0].scene_name, cell_objects)
    return cell

'''
Descriptions for Poses
'''

'''
Describes a pose based on the closest objects next to it.
CARE: the objects are not necessarily visible from e/o positions
Directions: +y <=> north, +x <=> east
TODO: add max-dist?
'''
def describe_pose(scene_objects: List[Object3D], pose: Pose, max_mentioned_objects=6):
    description = []

    object_distances = [obj.center - pose.eye for obj in scene_objects]
    object_distances = np.linalg.norm(object_distances, axis=1)
    closest_indices = np.argsort(object_distances)

    mentioned_objects = [scene_objects[idx] for idx in closest_indices[0:max_mentioned_objects]]

    for obj in mentioned_objects:
        obj2pose = pose.eye - obj.center # e.g. "The pose is south of a car."
        if abs(obj2pose[0])>=abs(obj2pose[1]) and obj2pose[0]>=0: direction='east'
        if abs(obj2pose[0])>=abs(obj2pose[1]) and obj2pose[0]<=0: direction='west'
        if abs(obj2pose[0])<=abs(obj2pose[1]) and obj2pose[1]>=0: direction='north'
        if abs(obj2pose[0])<=abs(obj2pose[1]) and obj2pose[1]<=0: direction='south'        

        do = DescriptionObject.from_object3d(obj, direction)
        description.append(do)
    
    return description
