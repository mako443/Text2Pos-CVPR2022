import numpy as np

class Object3d:
    def __init__(self, xyz, rgb, label, id):
        self.xyz = xyz
        self.rgb = rgb
        self.label = label
        self.id = id

    def apply_downsampling(self, indices):
        self.xyz = self.xyz[indices]
        self.rgb = self.rgb[indices]

    @classmethod
    def merge(cls, obj1, obj2):
        assert obj1.label==obj2.label and obj1.id==obj2.id
        return Object3d(
            np.vstack((obj1.xyz, obj2.xyz)),
            np.vstack((obj1.rgb, obj2.rgb)),
            obj1.label,
            obj1.id
        )