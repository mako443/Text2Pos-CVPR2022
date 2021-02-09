import numpy as np
import torch
import torch.nn
from torch.utils.data import Dataset

from torch_geometric.data import Data, DataLoader

'''
TODO:
- ok w/ random features?
- ok to h-stack description embeddings?
- convert to "I'm left of the car, house and fountain and right of the bush, tree and bench"?
'''

EMBED_DIM = 16

class MockGraphData(Dataset):
    def __init__(self):
        self.object_names = ('red', 'green', 'blue', 'yellow')
        self.object_embed = torch.nn.Embedding(4, EMBED_DIM) 
        with torch.no_grad():
            self.object_nodes = self.object_embed(torch.arange(4))

        self.object_positions = torch.tensor([(0,1), (1,0), (0,-1), (-1,0)], dtype=torch.float)
        self.num_objects = len(self.object_nodes)
        self.object_edges = torch.tensor([  [0,0,0,1,1,1,2,2,2,3,3,3],
                                            [1,2,3,0,2,3,0,1,3,0,1,2]], dtype=torch.long)

        self.description_edges  = torch.tensor([[0,0,0,1,1,1,2,2,2,3,3,3],
                                                [1,2,3,0,2,3,0,1,3,0,1,2]], dtype=torch.long)
        self.description_embed = torch.nn.Embedding(16, EMBED_DIM) #4x4 variations

    def __len__(self):
        return 4 #4 orientations

    def __getitem__(self, idx):
        idx=0
        with torch.no_grad():
            if idx==0: description_nodes = self.description_embed(torch.arange(4)+0)
            if idx==1: description_nodes = self.description_embed(torch.arange(4)+4)
            if idx==2: description_nodes = self.description_embed(torch.arange(4)+8)
            if idx==3: description_nodes = self.description_embed(torch.arange(4)+12)

        nodes = torch.cat((self.object_nodes, description_nodes))
        edges = torch.cat((self.object_edges, self.description_edges+self.num_objects), dim=-1)
        
        positive_objects = (torch.arange(4) + idx)%4
        positive_edges = torch.stack((positive_objects, torch.arange(4)+self.num_objects)) #These edges are only needed 1-dimensional (dot-product is symmetric)

        negative_edges = sample_negative_edges(positive_edges, 4, 4)

        return Data(x=nodes, edge_index=edges, edge_index_positive=positive_edges, edge_index_negative=negative_edges)

'''
Sample an edge from a random object to every description, makes sure true-positives are not sampled
Assumes positive edges as [obj, obj, obj; desc, desc, desc] - object-indices in top row, description indices in bottom row
'''
def sample_negative_edges(positive_edges, num_objects, num_descriptions):
    num_edges = positive_edges.size(1)
    assert num_edges==num_descriptions #There should be a true match for every description
    
    #Initialize
    negative_edges = torch.zeros((2, num_edges), dtype=torch.long)
    negative_edges[1,:] = positive_edges[1,:].clone()
    for col in range(num_edges):
        negative_edges[0, col] = int(np.random.choice([i for i in range(num_objects) if i!=positive_edges[0,col]]))

    return negative_edges

if __name__ == "__main__":
    data = MockGraphData()
    d = data[0]
    loader = DataLoader(data, batch_size=2, shuffle=False)
    b = next(iter(loader))
    
