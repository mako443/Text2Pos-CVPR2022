import numpy as np

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
    'building': 500,
    'pole': 500,
    'traffic light': 500,
    'traffic sign': 500,
    'garage': 500,
    'stop': 500,
    'smallpole': 500,
    'lamp': 500,
    'trash bin': 500,
    'vending machine': 500,
    'box': 500,
}

STUFF_CLASSES = ['sidewalk', 'road', 'parking', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'vegetation', 'terrain']

LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

COLORS = np.array([
    [0, 0, 0],
    [128, 128, 128],
    [255, 255, 255],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255]
])
COLOR_NAMES = [
    'black',
    'grey',
    'white',
    'red',
    'green',
    'blue',
    'yellow',
    'purple',
    'turquoise'
]