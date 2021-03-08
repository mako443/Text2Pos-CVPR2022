import numpy as np
import os
import os.path as osp
import pickle
import cv2

from datapreparation.imports import Object3D, ViewObject, Pose, DescriptionObject, calc_angle_diff
from datapreparation.drawing import draw_objects_poses, draw_objects_poses_viewObjects, draw_objects_poses_descriptions, draw_viewobjects, draw_objects_objectDescription, combine_images
from datapreparation.drawing import draw_cells
from datapreparation.descriptions import describe_pose, get_text_description, describe_object, describe_cell

import sys
sys.path.append('/home/imanox/Documents/Text2Image/Semantic3D-Net')
# sys.path.append('/usr/stud/kolmet/thesis/semantic3d')
from semantic.imports import ClusteredObject as ClusteredObject_S3D, ViewObject as ViewObject_S3D, COLORS, COLOR_NAMES
from graphics.imports import CLASSES_COLORS, Pose as Pose_S3D

'''
Module to load the Semantic3D clustered objects and view_objects
'''
def convert_s3d_data(path_pcd, path_images, split, scene_name):
    objects = pickle.load(open(osp.join(path_pcd, f'{scene_name}.objects.pkl'), 'rb'))
    objects = [Object3D.from_clustered_object_s3d(o) for o in objects]

    poses = pickle.load(open(osp.join(path_images, split, scene_name, 'poses_rendered.pkl'), 'rb'))
    poses = {k: Pose.from_pose_s3d(p) for k,p in poses.items()}
    
    view_objects = pickle.load(open(osp.join(path_images, split, scene_name, 'view_objects.pkl'), 'rb'))
    view_objects = {k: [ViewObject.from_view_object_s3d(v, convert_color_s3d(v.color)) for v in vos] for k,vos in view_objects.items()}

    return objects, poses, view_objects

def convert_s3d_data_objectsOnly(path_pcd, path_images, split, scene_name):  
    objects = pickle.load(open(osp.join(path_pcd, f'{scene_name}.objects.pkl'), 'rb'))
    objects = [Object3D.from_clustered_object_s3d(o) for o in objects]  
    return objects    


#Removes all but the <count> biggest objects of each class from the list, also removes their correspondences from the view-object lists
#Alternative: bigger clustering
def reduce_objects(objects, view_objects=None, count=16):
    reduced_objects = []
    for object_class in CLASSES_COLORS.keys():
        class_objects = [o for o in objects if o.label==object_class]
        class_objects = sorted(class_objects, key= lambda o: np.abs(np.max(o.points_w[:,0]) - np.min(o.points_w[:,0])) * np.abs(np.max(o.points_w[:,1]) - np.min(o.points_w[:,1])), reverse=True)
        reduced_objects.extend(class_objects[0:count])

    reduced_ids = [o.id for o in reduced_objects]

    if view_objects is None:
        return reduced_objects

    #Also remove the objects from the view-objects
    reduced_view_objects = {}
    for k in view_objects.keys():
        reduced_view_objects[k] = [vo for vo in view_objects[k] if vo.id in reduced_ids]

    return reduced_objects, reduced_view_objects

def convert_color_s3d(color_rgb):
    dists = np.linalg.norm(COLORS-color_rgb, axis=1)
    return COLOR_NAMES[np.argmin(dists)]

def describe_objects(scene_objects):
    all_descriptions, all_texts, all_hints = [], [], []
    for idx in range(len(scene_objects)):
        description, text, hints = describe_object(scene_objects, idx, max_mentioned_objects=5)
        all_descriptions.append(description)
        all_texts.append(text)
        all_hints.append(hints)

    return all_descriptions, all_texts, all_hints

def describe_cells(scene_objects, cell_size=25):
    object_bboxes = np.array([obj.aligned_bbox for obj in scene_objects]) #[x, y,z , wx, wh, wz]
    object_mins = object_bboxes[:, 0:3]
    object_maxs = object_bboxes[:, 0:3] + object_bboxes[:, 3:6]
    scene_min, scene_max = np.min(object_mins, axis=0), np.max(object_maxs, axis=0)
    
    cells = []
    best_cell = {'objects': []}
    for cell_x in range(int(scene_min[0]), int(scene_max[0]), cell_size):
        for cell_y in range(int(scene_min[1]), int(scene_max[1]), cell_size):
            cell_bbox = np.array([cell_x, cell_y, cell_x + cell_size, cell_y + cell_size])
            cell_data = describe_cell(scene_objects, cell_bbox)
            
            if cell_data is not None:
                cells.append(cell_data)
                if len(cell_data['objects']) > len(best_cell['objects']):
                    best_cell = cell_data

    return cells, best_cell

if __name__ == "__main__":
    path_pcd = 'data/numpy_merged/'
    path_images = 'data/pointcloud_images_o3d_merged_occ/'
    scene_name = 'bildstein_station1_xyz_intensity_rgb'
    output_dir = 'data/semantic3d'

    '''
    Creating cell-data -> need objects only
    '''
    if True:
        objects = convert_s3d_data_objectsOnly(path_pcd, path_images, 'train', scene_name)
        objects = [o for o in objects if o.label in ['high vegetation', 'low vegetation', 'buildings', 'hard scape', 'cars']]
        objects = reduce_objects(objects)
        
        cells, best_cell = describe_cells(objects)
        mean_cell_objects = np.mean([len(cell['objects']) for cell in cells])   

        pickle.dump(objects,      open(osp.join(output_dir,'train', scene_name, 'objects.pkl'), 'wb'))
        pickle.dump(cells,        open(osp.join(output_dir,'train', scene_name, 'cell_object_descriptions.pkl'), 'wb'))
        print(f'Saved {len(objects)} objects and {len(cells)} cells with {mean_cell_objects:0.2f} avg. objects to {osp.join(output_dir,"train", scene_name)}')
        
        cell_idx = np.argmax([len(cell['objects']) for cell in cells]) 
        img = cv2.flip(draw_cells(objects, cells, highlight_idx=cell_idx), 0)
        cv2.imwrite("cells.png", img)   
        quit()



    '''
    Creating all data (needs renderings)
    '''
    objects, poses, view_objects = convert_s3d_data(path_pcd, path_images, 'train', scene_name)
    #Remove stuff classes (at least for now) and retain only the k largest objects of each class
    objects = [o for o in objects if o.label in ['high vegetation', 'low vegetation', 'buildings', 'hard scape', 'cars']]
    objects, view_objects = reduce_objects(objects, view_objects)

    cells, best_cell = describe_cells(objects)
    mean_cell_objects = np.mean([len(cell['objects']) for cell in cells])

    descriptions, texts, hints = describe_objects(objects)

    pickle.dump(objects,      open(osp.join(output_dir,'train', scene_name, 'objects.pkl'), 'wb'))
    pickle.dump(descriptions, open(osp.join(output_dir,'train', scene_name, 'list_object_descriptions.pkl'), 'wb'))
    pickle.dump(texts,        open(osp.join(output_dir,'train', scene_name, 'text_object_descriptions.pkl'), 'wb'))
    pickle.dump(hints,        open(osp.join(output_dir,'train', scene_name, 'hint_object_descriptions.pkl'), 'wb'))
    pickle.dump(cells,        open(osp.join(output_dir,'train', scene_name, 'cell_object_descriptions.pkl'), 'wb'))
    print(f'Saved {len(objects)} objects, {len(descriptions)} descriptions, {len(texts)} texts and {len(cells)} cells with {mean_cell_objects:0.2f} avg. objects to {osp.join(output_dir,"train", scene_name)}')
    print()

    idx = np.random.randint(len(descriptions))
    print(texts[idx])
    print(hints[idx])
    img = cv2.flip(draw_objects_objectDescription(objects, descriptions[idx]), 0)
    cv2.imwrite("object_description.jpg", img)

    cell_idx = np.argmax([len(cell['objects']) for cell in cells]) 
    img = cv2.flip(draw_cells(objects, cells, highlight_idx=cell_idx), 0)
    cv2.imwrite("cells.png", img)   

    quit()

    #####

    # key = '147.png'
    description_lengths = []
    out_poses, out_descriptions, out_descriptionTexts = {}, {}, {}
    for key in poses.keys():
        pose = poses[key]

        description = describe_pose(view_objects, poses, key)
        description_text = get_text_description(description)

        out_poses[key] = pose
        out_descriptions[key] = description
        out_descriptionTexts[key] = description_text
        
        description_lengths.append(len(description))

    pickle.dump(objects, open(osp.join(output_dir,'train', scene_name, 'objects.pkl'), 'wb'))
    pickle.dump(out_poses, open(osp.join(output_dir,'train', scene_name, 'poses.pkl'), 'wb'))
    pickle.dump(out_descriptions, open(osp.join(output_dir,'train', scene_name, 'pose_descriptions.pkl'), 'wb'))
    pickle.dump(out_descriptionTexts, open(osp.join(output_dir,'train', scene_name, 'pose_description_texts.pkl'), 'wb'))
    print('Avg. objects per description: ', np.mean(description_lengths))

    quit()

    print(description_text)
    img = cv2.imread(osp.join(path_images, 'train', scene_name, 'rgb', key))
    img_vo = draw_viewobjects(img, view_objects[key])

    cv2.imshow("0", img_vo); #cv2.waitKey()

    img_pose = cv2.flip(draw_objects_poses_viewObjects(objects, poses, view_objects, (key,)), 0)
    cv2.imshow("1", img_pose); #cv2.waitKey()

    img_do = cv2.flip(draw_objects_poses_descriptions(objects, (pose,), (description,)), 0)
    cv2.imshow("2", img_do); cv2.waitKey()

    c = combine_images((img_vo, img_pose, img_do))
    cv2.imwrite("text-description.jpg", c)
    
