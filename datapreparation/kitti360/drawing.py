from typing import List
import numpy as np
import cv2
from datapreparation.kitti360.imports import Object3d, Cell, Description
from datapreparation.kitti360.utils import CLASS_TO_COLOR

try:
    import pptk
except:
    pass

def show_pptk(xyz, rgb):
    viewer = pptk.viewer(xyz)
    if isinstance(rgb, np.ndarray):
        viewer.attributes(rgb.astype(np.float32))
    else:
        attributes = [x.astype(np.float32) for x in rgb]
        viewer.attributes(*attributes)

    viewer.set(point_size=0.1)    

    return viewer

# Use scale=100 for cell-objects
def show_objects(objects: List[Object3d], scale=1.0):
    num_points = np.sum([len(obj.xyz) for obj in objects])
    xyz = np.zeros((num_points, 3), dtype=np.float)
    rgb1 = np.zeros((num_points, 3), dtype=np.uint8)
    rgb2 = np.zeros((num_points, 3), dtype=np.uint8)
    rgb3 = np.zeros((num_points, 3), dtype=np.uint8)
    offset = 0
    for obj in objects:
        rand_color = np.random.randint(low=0, high=256, size=3)
        c = CLASS_TO_COLOR[obj.label]
        # xyz = np.vstack((xyz, obj.xyz))
        # rgb1 = np.vstack((rgb1, np.ones((len(obj.xyz), 3))*rand_color ))
        # rgb2 = np.vstack((rgb2, obj.rgb))
        # rgb3 = np.vstack((rgb3, np.ones((len(obj.xyz), 3))*np.array(c) ))
        xyz[offset : offset+len(obj.xyz)] = obj.xyz
        rgb1[offset : offset+len(obj.xyz)] = np.ones((len(obj.xyz), 3))*rand_color
        rgb2[offset : offset+len(obj.xyz)] = obj.rgb * 255
        rgb3[offset : offset+len(obj.xyz)] = np.ones((len(obj.xyz), 3))*np.array(c)
        offset += len(obj.xyz)
    return show_pptk(xyz*scale, [rgb1 / 255.0, rgb2 / 255.0, rgb3 / 255.0])

def plot_cell(cell: Cell, scale=1024, use_rbg=False):
    img = np.zeros((scale, scale, 3), dtype=np.uint8)
    # Draw points of each object
    for obj in cell.objects:
        if obj.label == 'pad':
            continue
        c = CLASS_TO_COLOR[obj.label]
        for i_point, point in enumerate(obj.xyz*scale):
            if use_rbg:
                c = tuple(np.uint8(obj.rgb[i_point] * 255))
            point = np.int0(point[0:2])
            cv2.circle(img, tuple(point), 1, (int(c[2]),int(c[1]),int(c[0])))
    # Draw pose
    point = np.int0(cell.pose[0:2]*scale)
    cv2.circle(img, tuple(point), 10, (0,0,255), thickness=3)
    # Draw lines
    objects_dict = {obj.id: obj for obj in cell.objects}
    for descr in cell.descriptions:
        target = np.int0(objects_dict[descr.object_id].closest_point[0:2]*scale)
        cv2.arrowedLine(img, tuple(point), tuple(target), (0,0,255), thickness=2)
    return cv2.flip(img, 0) # Flip for correct north/south
