import numpy as np
import cv2

from datapreparation.imports import Object3D, ViewObject, Pose, CLASSES_COLORS

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

def draw_objects_poses(objects, poses, draw_arrows=True):
    scale, min_x, max_x, min_y, max_y = get_scale(objects)
    img = np.zeros((int(max_y-min_y)*scale, int(max_x-min_x)*scale, 3), np.uint8)

    for o in objects:
        points = (o.points_w[:,0:2] - np.array((min_x,min_y)))*scale
        rect = cv2.minAreaRect(points.astype(np.float32))
        box = np.int0(cv2.boxPoints(rect))
        c = CLASSES_COLORS[o.label]
        _ = cv2.drawContours(img,[box],0,(c[2],c[1],c[0]),thickness=3)  

    if type(poses)==dict: poses = list(poses.values())
    for p in poses:
        center = (p.eye[0:2] - np.array((min_x, min_y)))*scale
        end= center+15*scale*np.array((-np.cos(p.phi), -np.sin(p.phi)))
        cv2.circle(img, (int(center[0]), int(center[1])), 6, (0,0,255), 4)
        if draw_arrows:
            cv2.arrowedLine(img, (int(center[0]), int(center[1])), (int(end[0]), int(end[1])), (0,0,255), 4)

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

def draw_objects_poses_descriptions(objects, poses, descriptions):
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

def draw_cells(objects, cells):
    scale, min_x, max_x, min_y, max_y = get_scale(objects)
    img = draw_objects_poses(objects, [])

    for cell in cells:
        cell_bbox = cell['bbox']
        cell_mean = 0.5*(cell_bbox[0:2] + cell_bbox[2:4])

        bbox = np.int0((cell['bbox'] - np.array((min_x, min_y, min_x, min_y))) * scale)
        cv2.rectangle(img, tuple(bbox[0:2]), tuple(bbox[2:4]), (255,255,255), thickness=2)

        for obj in cell['objects']:
            center = obj.center_in_cell[0:2] + cell_mean
            center = np.int0((center - np.array((min_x, min_y))) * scale)
            c = CLASSES_COLORS[obj.label]
            cv2.circle(img, tuple(center), 3, (c[2], c[1], c[0]), thickness=2)

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


