from typing import List
import numpy as np
import pptk
from datapreparation.kitti360.imports import Object3d
from datapreparation.kitti360.utils import CLASS_TO_COLOR

def show_pptk(xyz, rgb):
    viewer = pptk.viewer(xyz)
    if isinstance(rgb, np.ndarray):
        viewer.attributes(rgb.astype(np.float32) / 255.0)
    else:
        attributes = [x.astype(np.float32)/ 255.0 for x in rgb]
        viewer.attributes(*attributes)

    viewer.set(point_size=0.1)    

    return viewer

def show_objects(objects: List[Object3d]):
    xyz = np.zeros((0,3), dtype=np.float)
    
    rgb1 = np.zeros((0,3), dtype=np.uint8)
    rgb2 = np.zeros((0,3), dtype=np.uint8)
    rgb3 = np.zeros((0,3), dtype=np.uint8)

    for obj in objects:
        obj_color = np.random.randint(low=0, high=256, size=3)
        xyz = np.vstack((xyz, obj.xyz))
        rgb1 = np.vstack((rgb1, np.ones((len(obj.xyz), 3))*obj_color ))
        rgb2 = np.vstack((rgb2, obj.rgb))
        c = CLASS_TO_COLOR[obj.label]
        rgb3 = np.vstack((rgb3, np.ones((len(obj.xyz), 3))*np.array(c) ))

    return show_pptk(xyz, [rgb1, rgb2, rgb3])