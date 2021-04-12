import numpy as np
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader

from analytical.utils import norm_angle

class MockData(Dataset):
    def __init__(self):
        self.object_classes = ['red', 'green', 'blue', 'yellow']
        self.object_positions = np.array([(0,1), (1,0), (0,-1), (-1,0)])

    def __len__(self):
        return 32

    # def __getitem_depr__(self, idx):
    #     data = {}
    #     data['object_classes'] = self.object_classes
    #     data['object_positions'] = self.object_positions

    #     data['description_classes'] = ['red', 'green', 'blue']
    #     data['description_directions'] = ['ahead', 'right', 'behind']
    #     return data

    #Manually checked ✓
    def __getitem__(self, idx):
        shuffle_indices = np.random.choice((0,1,2,3), size=4, replace=False)
        object_positions = self.object_positions[shuffle_indices]
        object_classes = np.random.choice(self.object_classes, size=4, replace=False)
        
        position = np.array((0,0))
        angle = np.random.choice((0,np.pi/2, -np.pi/2, np.pi)) # 0 is upwards (y direction), 

        description_directions = ['ahead', 'right', 'left']
        description_angles = (0, np.pi/2, -np.pi/2)
        description_classes = []
        match_indices = [] #For each element in description, remember which object it points to
        for i, description_angle in enumerate(description_angles):
            target_angle = norm_angle(angle+description_angle)
            target_position = np.array((np.sin(target_angle), np.cos(target_angle)))
            position_differences = np.linalg.norm(object_positions-target_position, axis=1)
            obj_index = np.argmin(position_differences)
            description_classes.append(object_classes[obj_index])
            match_indices.append(obj_index)

        #Shuffle the description
        shuffle_indices = np.random.choice((0,1,2), size=3, replace=False)
        description_directions = [description_directions[i] for i in shuffle_indices]
        description_classes = [description_classes[i] for i in shuffle_indices]
        match_indices = [match_indices[i] for i in shuffle_indices]

        return {'object_positions':object_positions,
                'object_classes':object_classes,
                'description_directions':description_directions,
                'description_classes':description_classes,
                'angles':angle,
                'match_indices':match_indices}


    #Custom collate_fn for this data | Where to do Padding?! (Here, in getitem, in separate method, in model?) TODO: pad in __getitem__ !
    #TODO: refer to specific names
    def collate_fn(samples):
        collated = {}
        for key in samples[0].keys():
            if type(samples[0][key])==list:
                collated[key] = [sample[key] for sample in samples]
            elif type(samples[0][key])==np.float64: #Angles
                collated[key] = np.array([sample[key] for sample in samples])
            elif type(samples[0][key])==np.ndarray:
                collated[key] = [sample[key] for sample in samples] #TODO: pad and stack
            else:
                raise Exception('Unexpected type in dataset! '+str(type(samples[0][key])))

        return collated

if __name__ == "__main__":
    dataset = MockData()
    data = dataset[0]        
    print(data['object_positions'])
    print(data['object_classes'])
    print()
    print(data['angles'])
    print(data['description_directions'])
    print(data['description_classes'])
    print()
    print(data['match_indices'])