from typing import List
import numpy as np
from sklearn.cluster import DBSCAN

from datapreparation.kitti360.imports import Object3d, Cell, Pose, DescriptionPoseCell, DescriptionBestCell
from datapreparation.kitti360.utils import STUFF_CLASSES

from copy import deepcopy

def get_mask(points, cell_bbox):
    mask = np.bitwise_and.reduce((
        points[:, 0] >= cell_bbox[0],
        points[:, 1] >= cell_bbox[1],
        points[:, 2] >= cell_bbox[2],
        points[:, 0] <= cell_bbox[3],
        points[:, 1] <= cell_bbox[4],
        points[:, 2] <= cell_bbox[5],
    ))   
    return mask 

def cluster_stuff_object(obj, stuff_min, eps=0.75):
    """ Perform DBSCAN cluster, thresh objects by points again
    """
    # cluster = DBSCAN(eps=1.5, min_samples=300, leaf_size=30, n_jobs=-1).fit(obj.xyz)
    cluster = DBSCAN(eps=eps, n_jobs=-1).fit(obj.xyz)
    clustered_objects = []

    for i, label_value in enumerate(range(0, np.max(cluster.labels_) + 1)):
        mask = cluster.labels_ == label_value
        if np.sum(mask) < stuff_min:
            continue

        c_obj = obj.mask_points(mask)
        clustered_objects.append(c_obj)

    return clustered_objects    

def create_synthetic_cell(bbox_w, area_objects: List[Object3d], min_objects=6, inside_fraction=1/3):  
    """Creates a synthetic cell for use in synthetic fine-localization training.
    Only threshes the objects inside/outside.
    Performs no normalization and does not re-set IDs.

    Args:
        bbox_w ([type]): [description]
        area_objects (List[Object3d]): [description]
        min_objects ([type], optional): [description]. Defaults to 6.
        inside_fraction ([type], optional): [description]. Defaults to 1/3.
    """
    cell_objects = [obj for obj in area_objects]
    # for obj in area_objects:
    #     mask = get_mask(obj.xyz, bbox_w)
    #     if np.sum(mask) / len(mask) < inside_fraction:
    #         continue
    #     cell_objects.append(obj)

    cell_size = np.max(bbox_w[3:6] - bbox_w[0:3])

    if len(cell_objects) < min_objects:
        return None

    return Cell(-1, "mock", cell_objects, cell_size, bbox_w)

def create_cell(cell_idx, scene_name, bbox_w, scene_objects: List[Object3d], min_objects=6, inside_fraction=1/3, stuff_min=500):
    cell_objects = []
    for obj in scene_objects:
        assert obj.id < 1e7

        mask = get_mask(obj.xyz, bbox_w)
        if obj.label in STUFF_CLASSES:
            if np.sum(mask) < stuff_min:
                continue

            cell_obj = obj.mask_points(mask)
            clustered_objects = cluster_stuff_object(cell_obj, stuff_min)
            cell_objects.extend(clustered_objects)
        else:
            if np.sum(mask) / len(mask) < inside_fraction:
                continue
            cell_objects.append(deepcopy(obj)) # Deep-copy to avoid changing scene-objects multiple times

    # Normalize objects based on the largest cell-edge to be ∈ [0, 1] (instance-objects can reach over edge)
    cell_size = np.max(bbox_w[3:6] - bbox_w[0:3])
    for obj in cell_objects:
        obj.xyz = (obj.xyz - bbox_w[0:3]) / cell_size

    # else: # If cell is synthetic, only copy objects and set cell-size
    #     cell_objects = scene_objects
    #     cell_size = np.max(bbox_w[3:6] - bbox_w[0:3])

    if len(cell_objects) < min_objects:
        return None        

    # Reset all ids
    for id, obj in enumerate(cell_objects):
        obj.id = id   

    return Cell(cell_idx, scene_name, cell_objects, cell_size, bbox_w)

def describe_pose_in_pose_cell(pose_w, cell: Cell, num_mentioned=6) -> List[DescriptionPoseCell]:
    # Assert pose is close to cell_center
    # assert np.allclose(pose_w, cell.get_center())
    assert len(cell.objects) >= num_mentioned, f'Only {len(cell.objects)} objects'

    # Norm pose
    pose = (pose_w - cell.bbox_w[0:3]) / cell.cell_size
    assert np.all(pose >= 0) and np.all(pose <= 1.0), f'{pose} {pose_w} {cell.bbox_w}'

    # Find the objects to describe
    descriptions = []
    distances = np.linalg.norm([obj.get_closest_point(pose) - pose for obj in cell.objects], axis=1)
    closest_indices = np.argsort(distances)

    # Create the descriptions
    mentioned_indices = closest_indices[0 : num_mentioned]

    for hint_idx, obj_idx in enumerate(mentioned_indices):
        obj = cell.objects[obj_idx]
        closest_point = obj.get_closest_point(pose)
        obj2pose = pose - closest_point
        if np.linalg.norm(obj2pose[0:2]) < 0.015:
            direction = 'on-top'
        else:
            if abs(obj2pose[0])>=abs(obj2pose[1]) and obj2pose[0]>=0: direction='east'
            if abs(obj2pose[0])>=abs(obj2pose[1]) and obj2pose[0]<=0: direction='west'
            if abs(obj2pose[0])<=abs(obj2pose[1]) and obj2pose[1]>=0: direction='north'
            if abs(obj2pose[0])<=abs(obj2pose[1]) and obj2pose[1]<=0: direction='south' 

        offset_center = pose - obj.get_center()
        offset_closest = pose - closest_point
        descriptions.append(DescriptionPoseCell(obj.id, obj.instance_id, obj.label, obj.get_color_rgb(), obj.get_color_text(), direction, offset_center, offset_closest, closest_point))

    return descriptions

def describe_pose_in_best_cell(pose_w: np.ndarray, pose_cell_descriptions: List[DescriptionPoseCell], cell: Cell) -> List[DescriptionBestCell]:
    # Assert cell is valid for this pose
    assert np.all(pose_w >= cell.bbox_w[0:3]) and np.all(pose_w <= cell.bbox_w[3:6]), f'{pose_w}, {cell.bbox_w}'
    assert len(cell.objects) >= len(pose_cell_descriptions), f'Only {len(cell.objects)} objects'    

    # Norm pose
    pose = (pose_w - cell.bbox_w[0:3]) / cell.cell_size
    assert np.all(pose >= 0) and np.all(pose <= 1.0), f'{pose} {pose_w} {cell.bbox_w}'

    best_cell_descriptions = []

    num_unmatched = 0
    matched_object_ids = []
    # Match the descriptions to the objects in the given cell
    for descr in pose_cell_descriptions:
        # Gather objects that have the correct instance_id and have not been matched yet.
        candidates = [obj for obj in cell.objects if (obj.instance_id == descr.object_instance_id and obj.id not in matched_object_ids)] 

        if len(candidates) == 0: # The description is not matched anymore in the best cell
            best_cell_descriptions.append(DescriptionBestCell.from_unmatched(descr))
            num_unmatched += 1            
        else: # The description is matched, convert it to best-matching candidate
            # Select the candidate with the best-matching closest_offset because that is less likely to have changed
            closest_offsets = np.array([pose - cand.get_closest_point(pose) for cand in candidates])[:, 0:2]
            best_idx = np.argmin(np.linalg.norm(closest_offsets - descr.offset_closest, axis=1))
            obj = candidates[best_idx]
            matched_object_ids.append(obj.id) # Prevent the object from being matched again
            
            closest_point = obj.get_closest_point(pose)
            # Currently only set in pose_cell and passed this way through
            # offset_center = pose - obj.get_center() 
            # offset_closest = pose - closest_point            

            # best_cell_descriptions.append(DescriptionBestCell.from_matched(descr, obj.id, offset_center, offset_closest, closest_point))
            best_cell_descriptions.append(DescriptionBestCell.from_matched(descr, obj.id, closest_point))

    return best_cell_descriptions, pose, num_unmatched
    

# TODO: shifted cells. 1) randomly shift cell around pose, 2) randomly shift, take objects from 2 threshs for missing hints (even necessary?)
def depr_describe_cell(bbox, scene_objects: List[Object3d], pose_w, scene_name, is_synthetic=False, inside_fraction=1/3, stuff_min=500, num_mentioned=6):
    """Create the cell using all the objects in the scene.
    Instance-objects are threshed in/outside the scene (all points are retained)
    Stuff-objects' points are threshed inside the cell, clustered and then saved with new IDs
    CARE: object-ids are completely re-set after gathering and can repeat across cells!

    Args:
        bbox: Cell bbox in world-coordinates
        scene_objects: Objects in scene
        pose_w: Pose in world-coordinates
        is_synthetic: Used for synthetic cell: skipping clustering and masking
    """
    if is_synthetic is False: # Mask, cluster and norm the scene objects
        cell_objects = []
        for obj in scene_objects:
            assert obj.id < 1e7

            mask = get_mask(obj.xyz, bbox)
            if obj.label in STUFF_CLASSES:
                if np.sum(mask) < stuff_min:
                    continue

                cell_obj = obj.mask_points(mask)
                clustered_objects = cluster_stuff_object(cell_obj, stuff_min)
                cell_objects.extend(clustered_objects)
            else:
                if np.sum(mask) / len(mask) < inside_fraction:
                    continue
                cell_objects.append(obj) # DEBUG: comment out

        # Normalize objects and pose based on the largest cell-edge to be ∈ [0, 1] (instance-objects can reach over edge)
        cell_size = np.max(bbox[3:6] - bbox[0:3])
        for obj in cell_objects:
            obj.xyz = (obj.xyz - bbox[0:3]) / cell_size
        pose = (pose_w - bbox[0:3]) / cell_size
        assert np.all(pose > 0) and np.all(pose < 1), f'{pose} {pose_w} {bbox}'
    else: # If cell is synthetic, only copy objects, cell-size and pose
        cell_objects = scene_objects
        cell_size = np.max(bbox[3:6] - bbox[0:3])
        pose = pose_w

    if len(cell_objects) < num_mentioned:
        return None

    # Reset all ids
    for id, obj in enumerate(cell_objects):
        obj.id = id + 1

    # Describe the post based on the clostest objects
    # Alternatives: describe in each direction, try to get many classes
    descriptions = []
    distances = np.linalg.norm([obj.get_closest_point(pose) - pose for obj in cell_objects], axis=1)
    closest_indices = np.argsort(distances)

    # mentioned_objects = [cell_objects[idx] for idx in closest_indices[0:num_mentioned]]
    mentioned_indices = [idx for idx in closest_indices[0 : num_mentioned]]
    # for obj in mentioned_objects:
    for hint_idx, obj_idx in enumerate(mentioned_indices):
        obj = cell_objects[obj_idx]

        obj2pose = pose - obj.closest_point # e.g. "The pose is south of a car."
        # if np.linalg.norm(obj2pose[0:2]) < 0.5 / cell_size: # Say 'on-top' if the object is very close (e.g. road), only calculated in x-y-plane!
        if np.linalg.norm(obj2pose[0:2]) < 0.015: # Say 'on-top' if the object is very close (e.g. road), only calculated in x-y-plane!
            direction = 'on-top'
        else:
            if abs(obj2pose[0])>=abs(obj2pose[1]) and obj2pose[0]>=0: direction='east'
            if abs(obj2pose[0])>=abs(obj2pose[1]) and obj2pose[0]<=0: direction='west'
            if abs(obj2pose[0])<=abs(obj2pose[1]) and obj2pose[1]>=0: direction='north'
            if abs(obj2pose[0])<=abs(obj2pose[1]) and obj2pose[1]<=0: direction='south' 

        descriptions.append(Description(obj.id, direction, obj.label, obj.get_color_rgb()))

    return Cell(scene_name, cell_objects, descriptions, pose, cell_size, pose_w, bbox)


    

