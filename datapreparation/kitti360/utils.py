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

LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}