import numpy as np

SCENE_NAMES = ('2013_05_28_drive_0000_sync','2013_05_28_drive_0002_sync','2013_05_28_drive_0003_sync','2013_05_28_drive_0004_sync','2013_05_28_drive_0005_sync','2013_05_28_drive_0006_sync','2013_05_28_drive_0007_sync','2013_05_28_drive_0009_sync','2013_05_28_drive_0010_sync')
SCENE_NAMES_TRAIN = ('2013_05_28_drive_0000_sync','2013_05_28_drive_0002_sync','2013_05_28_drive_0004_sync','2013_05_28_drive_0006_sync','2013_05_28_drive_0007_sync','2013_05_28_drive_0010_sync')
SCENE_NAMES_TEST = ('2013_05_28_drive_0003_sync', '2013_05_28_drive_0005_sync', '2013_05_28_drive_0009_sync')

SCENE_SIZES = {
    '2013_05_28_drive_0000_sync': [ 735, 1061,   30],
    '2013_05_28_drive_0002_sync': [ 952, 1313,   89],
    '2013_05_28_drive_0003_sync': [ 713,  922,   34],
    '2013_05_28_drive_0004_sync': [1302, 2003,   60],
    '2013_05_28_drive_0005_sync': [ 801,  999,   51],
    '2013_05_28_drive_0006_sync': [ 881, 1004,   80],
    '2013_05_28_drive_0007_sync': [3049, 1989,   52],
    '2013_05_28_drive_0009_sync': [ 615, 1113,   26],
    '2013_05_28_drive_0010_sync': [1560, 1445,   29],
}

CLASS_TO_INDEX = {
    'building':0,
    'pole':1,
    'traffic light':2,
    'traffic sign':3,
    'garage':4,
    'stop':5,
    'smallpole':6,
    'lamp':7,
    'trash bin':8,
    'vending machine':9,
    'box':10,
    'road': 11,
    'sidewalk':12,
    'parking':13,
    'wall': 14,
    'fence': 15,
    'guard rail': 16,
    'bridge': 17,
    'tunnel': 18,
    'vegetation': 19,
    'terrain': 20,
    'pad': 21,
}

CLASS_TO_LABEL = {
    'building':11,
    'pole':17,
    'traffic light':19,
    'traffic sign':20,
    'garage':34,
    'stop':36,
    'smallpole':37,
    'lamp':38,
    'trash bin':39,
    'vending machine':40,
    'box':41,
    'road': 7,
    'sidewalk':8,
    'parking':9,
    'wall': 12,
    'fence': 13,
    'guard rail': 14,
    'bridge': 15,
    'tunnel': 16,
    'vegetation': 21,
    'terrain': 22,
}

CLASS_TO_COLOR = {
    'building': ( 70, 70, 70),
    'pole': (153,153,153),
    'traffic light': (250,170, 30),
    'traffic sign': (220,220,  0),
    'garage': ( 64,128,128),
    'stop': (150,120, 90),
    'smallpole': (153,153,153),
    'lamp': (0,   64, 64),
    'trash bin': (0,  128,192),
    'vending machine': (128, 64,  0),
    'box': (64,  64,128),
    'sidewalk': (244, 35,232),
    'road': (128, 64,128),
    'parking': (250,170,160),
    'wall': (102,102,156),
    'fence': (190,153,153),
    'guard rail': (180,165,180),
    'bridge': (150,100,100),
    'tunnel': (150,120, 90),
    'vegetation': (107,142, 35),
    'terrain': (152,251,152), 
    '_pose': (255, 255, 255)   
}

CLASS_TO_MINPOINTS = {
    'building': 250,
    'pole': 25,
    'traffic light': 25,
    'traffic sign': 25,
    'garage': 250,
    'stop': 25,
    'smallpole': 25,
    'lamp': 25,
    'trash bin': 25,
    'vending machine': 25,
    'box': 25,
    'sidewalk': 1000,
    'road': 1000,
    'parking': 1000,
    'wall': 250,
    'fence': 250,
    'guard rail': 250,
    'bridge': 1000,
    'tunnel': 1000,
    'vegetation': 250,
    'terrain': 250, 
    '_pose': 25   
}

CLASS_TO_VOXELSIZE = {
    'building': 0.25,
    'pole': None,
    'traffic light': None,
    'traffic sign': None,
    'garage': 0.125,
    'stop': None,
    'smallpole': None,
    'lamp': None,
    'trash bin': None,
    'vending machine': None,
    'box': None,
    'sidewalk': 0.25,
    'road': 0.25,
    'parking': 0.25,
    'wall': 0.125,
    'fence': 0.125,
    'guard rail': 0.125,
    'bridge': 0.25,
    'tunnel': 0.25,
    'vegetation': 0.25,
    'terrain': 0.25, 
    '_pose': None   
}

STUFF_CLASSES = ['sidewalk', 'road', 'parking', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'vegetation', 'terrain']

LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

# COLORS = np.array([
#     [0, 0, 0],
#     [128, 128, 128],
#     [255, 255, 255],
#     [255, 0, 0],
#     [0, 255, 0],
#     [0, 0, 255],
#     [255, 255, 0],
#     [255, 0, 255],
#     [0, 255, 255]
# ])
# COLOR_NAMES = [
#     'black',
#     'grey',
#     'white',
#     'red',
#     'green',
#     'blue',
#     'yellow',
#     'purple',
#     'turquoise'
# ]

COLORS = np.array([
       [ 47.2579917 ,  49.75368454,  42.4153065 ],
       [136.32696657, 136.95241796, 126.02741229],
       [ 87.49822126,  91.69058836,  80.14558512],
       [213.91030679, 216.25033052, 207.24611073],
       [110.39218852, 112.91977458, 103.68638249],
       [ 27.47505158,  28.43996795,  25.16840296],
       [ 66.65951839,  70.22342483,  60.20395996],
       [171.00852191, 170.05737735, 155.00130334]
    ]) / 255.0

# COLOR_NAMES = ['color-0', 'color-1', 'color-2', 'color-3', 'color-4', 'color-5', 'color-6', 'color-7']
'''
Note that these names are not completely precise as the fitted colors are mostly gray-scale.
However, the models just learn them as words without deeper meaning, so they don't have a more complex effect.
'''
COLOR_NAMES = ['dark-green', 'gray', 'gray-green', 'bright-gray', 'gray', 'black', 'green', 'beige'] 

from scipy.spatial.distance import cdist
dists = cdist(COLORS, COLORS)