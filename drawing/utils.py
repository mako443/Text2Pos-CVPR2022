import numpy as np
import cv2

import sys
sys.path.append('/home/imanox/Documents/Text2Image/Semantic3D-Net')
import semantic.imports
from graphics.imports import CLASSES_COLORS, Pose

#CARE: angle might disagree w/ rendered images!
def plot_objects(objects, poses=[], pose_descriptions=[]):
    scale= 7

    min_x= np.min([np.min(o.points_w[:,0]) for o in objects])
    max_x= np.max([np.max(o.points_w[:,0]) for o in objects])
    min_y= np.min([np.min(o.points_w[:,1]) for o in objects])
    max_y= np.max([np.max(o.points_w[:,1]) for o in objects])

    img= np.zeros((int(max_y-min_y)*scale, int(max_x-min_x)*scale, 3), np.uint8)
    for o in objects:
        points= (o.points_w[:,0:2] - np.array((min_x,min_y)))*scale
        rect=cv2.minAreaRect(points.astype(np.float32))
        box=np.int0(cv2.boxPoints(rect))
        c= CLASSES_COLORS[o.label]
        _=cv2.drawContours(img,[box],0,(c[2],c[1],c[0]),thickness=3)

    for p in poses:
        center= np.array((p.eye[0]-min_x, p.eye[1]-min_y))*scale
        end= center+5*scale*np.array((np.cos(p.phi), np.sin(p.phi)))
        cv2.circle(img, (int(center[0]), int(center[1])), 2*scale, (0,0,255), scale)
        cv2.line(img, (int(center[0]), int(center[1])), (int(end[0]), int(end[1])), (0,0,255), 4)

    for i in range(len(pose_descriptions)):
        p= poses[i]
        center= np.array((p.eye[0]-min_x, p.eye[1]-min_y))*scale
        for _,target in pose_descriptions[i]:
            target= (np.mean(target.points_w,axis=0)[0:2]-(min_x,min_y)) * scale
            cv2.line(img, (int(center[0]), int(center[1])), (int(target[0]), int(target[1])), (200,50,150), 2)

    return img

def plot_ransac_guess(objects, center_estimate, ori_estimate, indices):
    print(indices)
    scale= 7

    min_x= np.min([np.min(o.points_w[:,0]) for o in objects])
    max_x= np.max([np.max(o.points_w[:,0]) for o in objects])
    min_y= np.min([np.min(o.points_w[:,1]) for o in objects])
    max_y= np.max([np.max(o.points_w[:,1]) for o in objects])

    #Plot objects
    img= np.zeros((int(max_y-min_y)*scale, int(max_x-min_x)*scale, 3), np.uint8)
    for o in objects:

        points= (o.points_w[:,0:2] - np.array((min_x,min_y)))*scale
        rect=cv2.minAreaRect(points.astype(np.float32))
        box=np.int0(cv2.boxPoints(rect))
        c= CLASSES_COLORS[o.label]
        _=cv2.drawContours(img,[box],0,(c[2],c[1],c[0]),thickness=3)  

    #Plot lines to anchor & verify objects
    center= np.array((center_estimate[0]-min_x, center_estimate[1]-min_y))*scale
    cv2.circle(img, (int(center[0]), int(center[1])), 2*scale, (0,0,255), scale)
    for idx in indices:
        o= np.mean(objects[idx].points_w,axis=0)
        end= np.array((o[0]-min_x, o[1]-min_y))*scale
        color= (0,0,255) if idx==indices[0] else (200,50,150)
        cv2.line(img, (int(center[0]), int(center[1])), (int(end[0]), int(end[1])), color, 2)

    return img    