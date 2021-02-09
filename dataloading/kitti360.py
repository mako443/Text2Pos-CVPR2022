import numpy as np
import cv2
import xml.etree.ElementTree as ET

#TODO: move to preparation

path = '/home/imanox/Documents/Text2Image/text2image-localization/data/kitti360/data_3d_bboxes/train/2013_05_28_drive_0000_sync.xml' 
CLASSES = ('bigPole', 'box', 'building', 'garage', 'lamp', 'smallPole', 'stop', 'trafficLight', 'trafficSign', 'trashbin', 'unknownConstruction', 'vendingmachine') #12 static classes
CLASSES_COLORS = {  'bigPole': (161, 0, 0),
                    'box': (0, 30, 255),
                    'building':  (168, 166, 50),
                    'garage': (124, 124, 125),
                    'lamp':  (0,96,255),
                    'smallPole': (255, 0, 0),
                    'stop': (204, 116, 116),
                    'trafficLight': (232, 232, 232),
                    'trafficSign': (61, 204, 45),
                    'trashbin': (143, 143, 143),
                    'unknownConstruction': (96, 96, 96),
                    'vendingmachine': (163, 0, 166)} #12 static classes

class KittiObject:
    def __init__(self, label, vertices, transform):
        self.label = label
        self.vertices = vertices
        self.transform = transform

        self.vertices = np.hstack((self.vertices, np.ones((len(self.vertices), 1))))
        self.vertices = self.vertices @ self.transform.T

    def get_bbox(self):
        b0 = np.min(self.vertices, axis=0)
        b1 = np.max(self.vertices, axis=0)
        return [b0[0], b0[1], b1[0], b1[1]]

tree = ET.parse(path)
root = tree.getroot()
xml_objects = root.getchildren()

objects = []

for i_obj, obj in enumerate(xml_objects):
    label = obj.find('label').text
    vertices = np.fromstring(obj.find('vertices').find('data').text.strip(), sep=" ").reshape((-1,3))
    transform = np.fromstring(obj.find('transform').find('data').text.strip(), sep=" ").reshape((4,4))

    objects.append(KittiObject(label, vertices, transform))

objects = [o for o in objects if o.label in CLASSES]    

object_bbox = np.array([o.get_bbox() for o in objects]) #[x0,y0,x1,y1]
min_x, min_y = np.min(object_bbox[:,0]), np.min(object_bbox[:,1])
object_bbox[:,0] -= min_x
object_bbox[:,2] -= min_x
object_bbox[:,1] -= min_y
object_bbox[:,3] -= min_y

img = np.zeros((2000,2000,3), np.uint8)
for bbox, obj in zip(object_bbox, objects):
    bbox*=2
    c = CLASSES_COLORS[obj.label]
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (c[2],c[1],c[0]), thickness=2 )

cv2.imshow("",img); cv2.waitKey()
cv2.imwrite("kitty-objects.jpg", img)