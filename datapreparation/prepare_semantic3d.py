import numpy as np
import os
import os.path as osp
import pickle
import cv2

from datapreparation.imports import Object3D, ViewObject, Pose, DescriptionObject, calc_angle_diff
from datapreparation.drawing import draw_objects_poses, draw_objects_poses_viewObjects, draw_objects_poses_descriptions, draw_viewobjects, draw_objects_objectDescription, combine_images
from datapreparation.descriptions import describe_pose, get_text_description, describe_object

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

#Removes all but the <count> biggest objects of each class from the list, also removes their correspondences from the view-object lists
#Alternative: bigger clustering
def reduce_objects(objects, view_objects, count=16):
    reduced_objects = []
    for object_class in CLASSES_COLORS.keys():
        class_objects = [o for o in objects if o.label==object_class]
        class_objects = sorted(class_objects, key= lambda o: np.abs(np.max(o.points_w[:,0]) - np.min(o.points_w[:,0])) * np.abs(np.max(o.points_w[:,1]) - np.min(o.points_w[:,1])), reverse=True)
        reduced_objects.extend(class_objects[0:count])

    reduced_ids = [o.id for o in reduced_objects]

    reduced_view_objects = {}
    for k in view_objects.keys():
        reduced_view_objects[k] = [vo for vo in view_objects[k] if vo.id in reduced_ids]

    return reduced_objects, reduced_view_objects

def convert_color_s3d(color_rgb):
    dists = np.linalg.norm(COLORS-color_rgb, axis=1)
    return COLOR_NAMES[np.argmin(dists)]

def describe_objects(scene_objects):
    all_descriptions, all_texts = [], []
    for idx in range(len(scene_objects)):
        description, text = describe_object(scene_objects, idx, max_mentioned_objects=5)
        all_descriptions.append(description)
        all_texts.append(text)

    return all_descriptions, all_texts

if __name__ == "__main__":
    path_pcd = 'data/numpy_merged/'
    path_images = 'data/pointcloud_images_o3d_merged_occ/'
    scene_name = 'sg27_station5_intensity_rgb'
    objects, poses, view_objects = convert_s3d_data(path_pcd, path_images, 'train', scene_name)
    output_dir = 'data/semantic3d'

    #Remove stuff classes (at least for now)
    objects = [o for o in objects if o.label in ['high vegetation', 'low vegetation', 'buildings', 'hard scape', 'cars']]
    objects, view_objects = reduce_objects(objects, view_objects)

    descriptions, texts = describe_objects(objects)

    pickle.dump(objects,      open(osp.join(output_dir,'train', scene_name, 'objects.pkl'), 'wb'))
    pickle.dump(descriptions, open(osp.join(output_dir,'train', scene_name, 'list_object_descriptions.pkl'), 'wb'))
    pickle.dump(texts,        open(osp.join(output_dir,'train', scene_name, 'text_object_descriptions.pkl'), 'wb'))
    print(f'Saved {len(objects)} objects, {len(descriptions)} descriptions and {len(texts)} texts to {osp.join(output_dir,"train", scene_name)}')

    idx = np.random.randint(len(descriptions))
    print(texts[idx])
    img = cv2.flip(draw_objects_objectDescription(objects, descriptions[idx]), 0)
    cv2.imwrite("object_description.jpg", img)

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
    
