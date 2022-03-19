import numpy as np
import cv2

import pickle
import os
import os.path as osp

import sys

sys.path.append("/home/imanox/Documents/Text2Image/Semantic3D-Net")
from semantic.imports import ViewObject, project_points, CORNERS
from graphics.imports import CLASSES_COLORS, Pose, IMAGE_WIDHT, IMAGE_HEIGHT

"""
Attempt to solve "discretized" correspondences via PnpRANSAC: doens't work ✖, pnp fails after rounding
"""

CORNERS = (
    (0.25, 0.25),
    (0.5, 0.25),
    (0.75, 0.25),
    (0.25, 0.5),
    (0.5, 0.5),
    (0.75, 0.5),
    (0.25, 0.75),
    (0.5, 0.75),
    (0.75, 0.75),
)


def calc_pose_err(eye, tvec):
    tvec = tvec.flatten()
    print(eye, (-tvec[2], tvec[0], tvec[1]))
    return np.float16(
        np.linalg.norm((-eye[0] - tvec[2], eye[1] - tvec[0], eye[2] - tvec[1]))
        / np.linalg.norm(eye)
    )


def round_image_points(image_points):
    corners = CORNERS * np.array((IMAGE_WIDHT, IMAGE_HEIGHT))
    dists = np.expand_dims(image_points, 1) - corners
    dists = np.linalg.norm(dists, axis=2)
    indices = np.argmin(dists, axis=1)
    return corners[indices]


root = "./data/pointcloud_images_o3d_merged/test/sg27_station2_intensity_rgb/"
view_objects = pickle.load(open(osp.join(root, "view_objects.pkl"), "rb"))
poses_rendered = pickle.load(open(osp.join(root, "poses_rendered.pkl"), "rb"))
classes = ("high vegetation", "low vegetation", "buildings", "cars")

shift = 200
plot = np.zeros((2 * shift, 2 * shift, 3), dtype=np.uint8)

idx = list(poses_rendered.keys())[0]
img = cv2.imread(osp.join(root, "rgb", idx))
pose = poses_rendered[idx]
vo = [v for v in view_objects[idx] if v.label in classes]

object_points = []
image_points = []
for v in vo:
    p = np.mean(v.points_w, axis=0)
    object_points.append(p)
    p = project_points(pose.I, pose.E, p.reshape((1, 3))).flatten()
    image_points.append(p[0:2])
    c = CLASSES_COLORS[v.label]
    cv2.circle(img, (int(p[0]), int(p[1])), 5, (c[2], c[1], c[0]), 3)

image_points_rounded = round_image_points(image_points)

retval, rvec, tvec0, inliers = cv2.solvePnPRansac(
    np.array(object_points), np.array(image_points), pose.I, distCoeffs=None
)
print(calc_pose_err(pose.eye, tvec0))
retval, rvec, tvec1, inliers = cv2.solvePnPRansac(
    np.array(object_points), np.array(image_points_rounded), pose.I, distCoeffs=None
)
print(calc_pose_err(pose.eye, tvec1))

# plot=cv2.resize(plot, (4*shift, 4*shift))
cv2.imshow("", img)
cv2.waitKey()
