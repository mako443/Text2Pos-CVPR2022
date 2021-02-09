import numpy as np
import cv2

ANGLES= {'infront': 0.0, 'right': np.pi/2, 'left': -np.pi/2, 'behind': np.pi}

#Bring angle to ∈[-pi,pi]
def norm_angle(angle):
    angle= np.array((angle,))
    angle= np.mod(angle,2*np.pi)
    angle[angle>np.pi]-= 2*np.pi
    if len(angle)==1: return angle[0]
    else: return angle

def describe_pose(objects, pose):
    description= []
    max_angle= np.pi/16 #Find object thats closer than 45° to direction
    for angle in list(ANGLES.keys()):
        target_angle= pose.phi+ANGLES[angle] # ∈[-pi,pi]

        directions= np.array([ np.mean(o.points_w, axis=0) for o in objects]) - pose.eye

        object_angles= np.arctan2(directions[:,1], directions[:,0]) #CARE: order and negative factor, currently all in image-coordinates!
        angle_differences= np.abs(norm_angle(object_angles) - norm_angle(target_angle))

        #Use all
        for idx in np.argwhere(angle_differences<max_angle).flatten():
            description.append((angle, objects[idx]))

        #Use only clostest
        # if np.min(angle_differences)<=max_angle:
        #     description.append((angle, objects[np.argmin(angle_differences)]))
            
    return description