import numpy as np
import cv2

import pickle
import os
import os.path as osp
from drawing.utils import plot_objects, plot_ransac_guess
from analytical.utils import describe_pose, ANGLES, norm_angle

import sys
sys.path.append('/home/imanox/Documents/Text2Image/Semantic3D-Net')
from semantic.imports import ViewObject, project_points, CORNERS, COLORS, COLOR_NAMES
from graphics.imports import CLASSES_COLORS, Pose, IMAGE_WIDHT, IMAGE_HEIGHT

'''
TODO
- if continued beyond 1st demo: clean and refactor!
'''

def discretize_color(color):
    color = np.array(color).reshape((-1,1,3))
    dists = np.linalg.norm(color-COLORS, axis =-1)
    indices = np.argmin(dists, axis =1)
    closest_colors = np.array(COLOR_NAMES)[indices]
    if len(closest_colors)==1:
        return closest_colors[0]
    else:
        return closest_colors

def localize_ransac(clustered_objects, description):
    object_classes = np.array([c.label for c in clustered_objects])
    object_centers = np.array([ np.mean(o.points_w, axis = 0) for o in clustered_objects])[:,0:2]
    object_colors = discretize_color([o.color for o in clustered_objects])

    best_error =  np.inf
    best_center =  None
    best_ori = None
    best_indices = None

    for iter in range(1000):
        #Sample 2 anchor objects (in-front, behind)
        anchor0_description_idx = np.random.choice([i for i in range(len(description)) if description[i][0]=='infront'])
        anchor1_description_idx = np.random.choice([i for i in range(len(description)) if description[i][0]=='behind'])
        anchor0_class = description[anchor0_description_idx][1].label
        anchor0_color = discretize_color(description[anchor0_description_idx][1].color)
        anchor1_class = description[anchor1_description_idx][1].label
        anchor1_color = discretize_color(description[anchor1_description_idx][1].color)

        anchor0_object_idx = np.random.choice([i for i in range(len(object_classes)) if object_classes[i]==anchor0_class and object_colors[i]==anchor0_color])
        anchor1_object_idx = np.random.choice([i for i in range(len(object_classes)) if object_classes[i]==anchor1_class and object_colors[i]==anchor1_color])

        center_estimate = 1/2*(object_centers[anchor0_object_idx]+object_centers[anchor1_object_idx])
        vector = object_centers[anchor0_object_idx]-object_centers[anchor1_object_idx]
        ori_estimate = np.arctan2(vector[1],vector[0])

        remaining_object_indices = np.array([i for i in range(len(object_classes)) if i not in (anchor0_object_idx, anchor1_object_idx)])
        remaining_description_indices = np.array([i for i in range(len(description)) if i not in (anchor0_description_idx, anchor1_description_idx)])

        #Verify using best-fitting objects for remaining descriptions
        angle_error = 0
        found_verify_indices = []
        for idx_verify in remaining_description_indices:
            angle_name, _= description[idx_verify]
            angle = ANGLES[angle_name]
            target_angle = norm_angle(ori_estimate+angle)

            object_vectors = object_centers-center_estimate
            object_angles = np.arctan2(object_vectors[:,1], object_vectors[:,0])
            angle_differences = np.abs(object_angles-target_angle)
            angle_differences = np.minimum(angle_differences, 2*np.pi-angle_differences) 

            angle_differences[ object_classes!=description[idx_verify][1].label ]= np.inf #Remove objects from false classes
            angle_differences[ object_colors !=discretize_color(description[idx_verify][1].color) ]= np.inf #Remove objects from false colors
            angle_differences[ found_verify_indices ] = np.inf #Remove already used objects
            angle_differences[ anchor0_object_idx ] = np.inf #Remove anchor objects
            angle_differences[ anchor1_object_idx ] = np.inf

            found_verify_indices.append(np.argmin(angle_differences))
            angle_error+= np.min(angle_differences)**2

        if angle_error< best_error:
            best_error, best_center, best_ori, best_indices = angle_error, center_estimate, ori_estimate, (anchor0_object_idx, anchor1_object_idx, *found_verify_indices)

        # img =plot_ransac_guess(clustered_objects, center_estimate, ori_estimate, (idx_a0, idx_a1, *found_verify_indices))
    # print(len(best_indices))
    # img =plot_ransac_guess(clustered_objects, best_center, best_ori, best_indices)
    # cv2.imshow("2",img); cv2.waitKey()

    return best_center, best_ori, best_indices

scene = 'sg27_station2_intensity_rgb'
clustered_objects = pickle.load(open(osp.join('./data/numpy_merged', f'{scene}.objects.pkl'), 'rb'))
root = f'./data/pointcloud_images_o3d_merged/test/{scene}'
view_objects = pickle.load(open(osp.join(root,'view_objects.pkl'), 'rb'))
poses_rendered = pickle.load(open(osp.join(root,'poses_rendered.pkl'), 'rb'))

classes =('high vegetation', 'low vegetation', 'buildings', 'cars', 'hard scape')
clustered_objects = [o for o in clustered_objects if o.label in classes]

# idx = '011.png'
# img = cv2.imread(osp.join(root,'rgb',idx))
# plot = plot_objects(clustered_objects, poses =(pose,), pose_descriptions =(description,))
# cv2.imshow("1",plot); cv2.waitKey()

center_errors, ori_errors = [], []
correctly_localized = []
for idx in ('011.png',):
    print(f'Idx: {idx}')
    pose = poses_rendered[idx]
    description = describe_pose(clustered_objects, pose)

    p_center, p_ori, p_indices = localize_ransac(clustered_objects, description)

    center_error = np.linalg.norm(pose.eye[0:2] - p_center)
    center_errors.append(center_error)
    ori_error = np.abs(pose.phi - p_ori)
    ori_error = np.minimum(ori_error, 2*np.pi-ori_error)
    ori_errors.append(ori_error)

    correctly_localized.append(center_error<=10 and ori_error<=np.pi/4)

    img = cv2.imread(osp.join(root,'rgb',idx))
    plot0 = plot_objects(clustered_objects, poses =(pose,), pose_descriptions =(description,))
    plot1 = plot_ransac_guess(clustered_objects, p_center, p_ori, p_indices)
    cv2.imshow("0",plot0)
    cv2.imshow("1",plot1)
    cv2.waitKey()
    cv2.imwrite("loc-gt.jpg", plot0)
    cv2.imwrite("loc-estimate.jpg", plot1)

print(np.mean(center_errors), np.mean(ori_errors))
print(np.mean(correctly_localized))
