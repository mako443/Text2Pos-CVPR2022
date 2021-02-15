import numpy as np
import cv2

#TODO: move these to the respective "prepare-X" modules
CLASSES_DICT={'unlabeled': 0, 'man-made terrain': 1, 'natural terrain': 2, 'high vegetation': 3, 'low vegetation': 4, 'buildings': 5, 'hard scape': 6, 'scanning artefacts': 7, 'cars': 8}
CLASSES_COLORS={'unlabeled': (255,255,255), 'man-made terrain': (60,30,30), 'natural terrain': (30,60,30), 'high vegetation': (120,255,120), 'low vegetation': (60,150,60), 'buildings': (255,255,0), 'hard scape': (0,255,255), 'scanning artefacts': (255,0,0), 'cars': (0,0,255)}
IMAGE_WIDHT=1620
IMAGE_HEIGHT=1080
DIRECTIONS = {'ahead': 0.0, 'behind': np.pi, 'right': -np.pi/2, 'left': np.pi/2 } #CARE: Directions go counter-clockwise
DIRECTIONS_COMPASS = {'north': 0.0, 'east': -np.pi/2, 'south': np.pi, 'west': np.pi/2}

COLOR_NAMES=('brightness-0','brightness-1','brightness-2','brightness-3','brightness-4','brightness-5','brightness-6','brightness-7')
COLORS=np.array([[0.15136254, 0.12655825, 0.12769653],
                [0.22413703, 0.19569607, 0.20007613],
                [0.29251393, 0.2693559 , 0.27813852],
                [0.35667216, 0.3498905 , 0.36508256],
                [0.45776146, 0.39058182, 0.38574897],
                [0.45337288, 0.46395565, 0.47795434],
                [0.52570801, 0.53530194, 0.56404256],
                [0.66988167, 0.6804131 , 0.71069241]])

class Object3D:
    @classmethod
    def from_clustered_object_s3d(cls, co):
        o = Object3D()
        o.scene_name = co.scene_name
        o.label = co.label
        o.id = co.clustered_object_id
        o.points_w = co.points_w
        o.color = co.color
        return o

    @property
    def center(self):
        return 1/2*(np.min(self.points_w[:, 0:2], axis=0) + np.max(self.points_w[:, 0:2], axis=0))

    def __repr__(self):
        return f'Object3D: {self.label} at {self.center}'

    @property
    def rotated_bbox(self):
        points = self.points_w[:,0:2]
        rect = cv2.minAreaRect(points.astype(np.float32))
        box = np.int0(cv2.boxPoints(rect))       
        return box 

class ViewObject:
    @classmethod
    def from_view_object_s3d(cls, vo, color_text):
        o = ViewObject()
        o.scene_name = vo.scene_name
        o.label = vo.label
        o.points_i = vo.points
        o.rect_i = vo.rect
        o.color_rgb = vo.color
        o.color_text = color_text
        o.id = vo.clustered_object_id
        return o

    #CARE: can't naively use rect_i for center
    @property
    def center(self):
        return 1/2*(np.min(self.points_i[:, 0:2], axis=0) + np.max(self.points_i[:, 0:2], axis=0))

class Pose:
    @classmethod
    def from_pose_s3d(cls, pose):
        p = Pose()
        p.eye = pose.eye
        p.forward = pose.forward
        p.phi = pose.phi
        p.E = pose.E
        return p

class DescriptionObject:
    @classmethod
    def from_view_object(cls, vo, direction):        
        d = DescriptionObject()
        d.direction = direction # None if the object is target?
        d.label = vo.label
        d.color_rgb = vo.color_rgb
        d.color_text = vo.color_text
        d.id = vo.id
        return d

    @classmethod
    def from_object3d(cls, o3d, direction):
        d = DescriptionObject()
        d.direction = direction
        d.label = o3d.label
        d.id = o3d.id
        d.color_rgb = o3d.color

        color_dists = np.linalg.norm(COLORS - o3d.color, axis=1)
        d.color_text = COLOR_NAMES[np.argmin(color_dists)]

        return d

    def __str__(self):
        
        return f'{self.color_text} {self.label} {self.direction}'

def calc_angle_diff(a,b):
    return np.abs(np.arctan2(np.sin(a-b), np.cos(a-b)))

