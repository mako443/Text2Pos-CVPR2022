import numpy as np
import cv2

from datapreparation.semantic3d.imports import Object3D, ViewObject, Pose, CLASSES_COLORS

'''
Module to draw objects, poses and pose-descriptions onto a 2D map. CARE: these images have to be x-flipped to be consistent with world coordinates!
'''
def get_scale(objects):
    scale = 7

    min_x = np.min([np.min(o.points_w[:,0]) for o in objects])
    max_x = np.max([np.max(o.points_w[:,0]) for o in objects])
    min_y = np.min([np.min(o.points_w[:,1]) for o in objects])
    max_y = np.max([np.max(o.points_w[:,1]) for o in objects]) 
    return scale, min_x, max_x, min_y, max_y    

def draw_objects_poses(objects, poses, draw_arrows=True, pose_descriptions=None):
    if pose_descriptions is not None:
        assert len(poses) == len(pose_descriptions)

    scale, min_x, max_x, min_y, max_y = get_scale(objects)
    img = np.zeros((int(max_y-min_y)*scale, int(max_x-min_x)*scale, 3), np.uint8)
    objects_dict = {o.id : o for o in objects}  

    for o in objects:
        points = (o.points_w[:,0:2] - np.array((min_x,min_y)))*scale
        rect = cv2.minAreaRect(points.astype(np.float32))
        box = np.int0(cv2.boxPoints(rect))
        c = CLASSES_COLORS[o.label]
        _ = cv2.drawContours(img,[box],0,(c[2],c[1],c[0]),thickness=3)  

    if type(poses)==dict: poses = list(poses.values())
    for i_p, p in enumerate(poses):
        center = np.int0((p.eye[0:2] - np.array((min_x, min_y)))*scale)
        # end= center+15*scale*np.array((-np.cos(p.phi), -np.sin(p.phi)))
        # if draw_arrows: cv2.arrowedLine(img, (int(center[0]), int(center[1])), (int(end[0]), int(end[1])), (0,0,255), 4)  
        cv2.circle(img, (int(center[0]), int(center[1])), 6, (0,0,255), 4)
        if pose_descriptions is not None:
            for do in pose_descriptions[i_p]:
                object3d = objects_dict[do.id]
                end = np.int0(( 0.5*(np.max(object3d.points_w[:, 0:2], axis=0) + np.min(object3d.points_w[:, 0:2], axis=0)) - np.array((min_x, min_y))) * scale)
                cv2.arrowedLine(img, (center[0], center[1]), (end[0], end[1]), (255,255,255), thickness=2)            
    return img

def draw_objects_poses_viewObjects(objects, poses, view_objects, keys):
    scale, min_x, max_x, min_y, max_y = get_scale(objects)
    img = draw_objects_poses(objects, [poses[k] for k in keys])
    objects_dict = {o.id : o for o in objects}

    for key in keys:
        pose = poses[key]
        vos = view_objects[key]
        for vo in vos:
            start = (pose.eye[0:2] - np.array((min_x, min_y)))*scale
            object3d = objects_dict[vo.id]
            end = ( 0.5*(np.max(object3d.points_w[:, 0:2], axis=0) + np.min(object3d.points_w[:, 0:2], axis=0)) - np.array((min_x, min_y))) * scale
            cv2.arrowedLine(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (155,155,155), 3)

    return img

def draw_objects_poses_descriptions_DEPRECATED(objects, poses, descriptions):
    assert len(poses)==len(descriptions)
    scale, min_x, max_x, min_y, max_y = get_scale(objects)
    img = draw_objects_poses(objects, poses)
    objects_dict = {o.id : o for o in objects}

    for pose, description in zip(poses, descriptions):
        for do in description:
            start = (pose.eye[0:2] - np.array((min_x, min_y)))*scale
            object3d = objects_dict[do.id]
            end = ( 0.5*(np.max(object3d.points_w[:, 0:2], axis=0) + np.min(object3d.points_w[:, 0:2], axis=0)) - np.array((min_x, min_y))) * scale
            c = (255,0,0) if do.direction=='ahead' else (255,255,255)
            cv2.arrowedLine(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255,255,255), 3)            

    return img

def draw_objects_poseDescription(objects, pose, description):
    scale, min_x, max_x, min_y, max_y = get_scale(objects)
    img = draw_objects_poses(objects, [])
    objects_dict = {o.id : o for o in objects}    

    center = np.int0((pose.eye[0:2] - np.array((min_x, min_y)))*scale)
    cv2.circle(img, (center[0], center[1]), 5, (0,0,255), thickness=2)
    for do in description:
        object3d = objects_dict[do.id]
        end = np.int0(( 0.5*(np.max(object3d.points_w[:, 0:2], axis=0) + np.min(object3d.points_w[:, 0:2], axis=0)) - np.array((min_x, min_y))) * scale)
        cv2.arrowedLine(img, (center[0], center[1]), (end[0], end[1]), (255,255,255), thickness=2)

    return img


def draw_objects_objectDescription(objects, description_list):
    scale, min_x, max_x, min_y, max_y = get_scale(objects)
    img = draw_objects_poses(objects, [])
    objects_dict = {o.id : o for o in objects}    

    for d_obj in description_list:
        if d_obj.direction is None:
            target_center = (objects_dict[d_obj.id].center[0:2] - np.array((min_x,min_y)))*scale

    for d_obj in description_list:
        if d_obj.direction is None:
            continue
        obj3d = objects_dict[d_obj.id]
        source_center = (objects_dict[d_obj.id].center[0:2] - np.array((min_x,min_y)))*scale
        cv2.arrowedLine(img, (int(target_center[0]), int(target_center[1])), (int(source_center[0]), int(source_center[1])), (128,128,0), thickness=3 )


    return img

def draw_cells(objects, cells, highlight_indices=[], poses=[], pose_descriptions=None):
    scale, min_x, max_x, min_y, max_y = get_scale(objects)
    img = draw_objects_poses(objects, poses, draw_arrows=False, pose_descriptions=pose_descriptions)

    for idx, cell in enumerate(cells):
        cell_bbox = cell.bbox
        cell_mean = 0.5*(cell_bbox[0:2] + cell_bbox[2:4])

        bbox = np.int0((cell.bbox - np.array((min_x, min_y, min_x, min_y))) * scale)
        c = (0,0,255) if idx in highlight_indices else (255,255,255)
        t = 4 if idx in highlight_indices else 1
        cv2.rectangle(img, tuple(bbox[0:2]), tuple(bbox[2:4] - np.array((5, 5),dtype=np.int)), c, thickness=t) # Subtract a few pixels to see different overlapping cells

        for obj in cell.objects:
            center = obj.center_in_cell[0:2] + cell_mean
            center = np.int0((center - np.array((min_x, min_y))) * scale)
            c = CLASSES_COLORS[obj.label]
            cv2.circle(img, tuple(center), 3, (c[2], c[1], c[0]), thickness=2)

    return img

# To be used with Semantic3dPosesDataset
def draw_retrieval(dataset, pose_idx, top_cell_indices, k=3):
    scale, min_x, max_x, min_y, max_y = get_scale(dataset.scene_objects)
    img = draw_objects_poses(dataset.scene_objects, dataset.poses[pose_idx:pose_idx+1], draw_arrows=False, pose_descriptions=dataset.pose_descriptions[pose_idx:pose_idx+1])
    pose = dataset.poses[pose_idx]
    pose = np.int0((pose.eye[0:2] - np.array((min_x, min_y)))*scale)
    
    # Draw the best cell in green
    cell = dataset.cells[dataset.best_cell_indices[pose_idx]]
    bbox = np.int0((cell.bbox - np.array((min_x, min_y, min_x, min_y))) * scale)
    cv2.rectangle(img, tuple(bbox[0:2]), tuple(bbox[2:4]), (0,255,0), thickness=2)

    # Draw the top retrievals in red
    for idx in top_cell_indices[0:k]:
            cell = dataset.cells[idx]
            bbox = np.int0((cell.bbox - np.array((min_x, min_y, min_x, min_y))) * scale)
            center = np.int0( 0.5*(bbox[0:2] + bbox[2:4]))
            offset = np.random.randint(-15, 15, size=4)
            cv2.rectangle(img, tuple(bbox[0:2] + offset[0:2]), tuple(bbox[2:4] + offset[2:4]), (0,0,255), thickness=2)
            cv2.line(img, tuple(pose), tuple(center), (0,0,255), thickness=2)

    return img

def draw_viewobjects(img, view_objects):
    for v in view_objects:
        box = np.int0(cv2.boxPoints(v.rect_i))
        color = CLASSES_COLORS[v.label]
        cv2.drawContours(img,[box],0, (color[2], color[1], color[0]),thickness=3)
    return img

def combine_images(images):
    h = np.max([i.shape[0] for i in images])
    w = np.sum([i.shape[1] for i in images])
    combined = np.zeros((h,w,3), dtype=np.uint8)
    current_w = 0
    for img in images:
        combined[0:img.shape[0], current_w:current_w+img.shape[1], :] = img
        current_w += img.shape[1]
        combined[:, current_w-3:current_w, :] = (255,255,255)
    return combined

