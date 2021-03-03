import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from models.modules import get_mlp, LanguageEncoder
from dataloading.semantic3d import Semantic3dObjectReferanceDataset

from training.args import parse_arguments
from training.plots import plot_metrics

'''
TODO:
- Use constant target -> still nor working ✓
- Check if error somewhere, check inputs, outputs: inputs + targets constant and same shape -> still not working, looks like dimension-distance too big? ✖
- Normalize all to [-1, 1] -> No ✖
'''

class OffsetPredict(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(OffsetPredict, self).__init__()

        self.mlp = nn.ModuleList([get_mlp([2, 8, 16, 32, 16, 8, 2]), ])

    def forward(self, positions):
        offsets = self.mlp[0](positions)
        return offsets


learning_rates = np.logspace(-2, -4, 5)
dict_losses = {lr: [] for lr in learning_rates}

targets = 50 * (0.5-torch.rand(2,6,2, requires_grad=False))

for lr in (0.1, 0.05, 0.01, 0.005):
    model = OffsetPredict(-1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset_train = Semantic3dObjectReferanceDataset('./data/numpy_merged/', './data/semantic3d', num_distractors=0, split='train')
    dataloader_train = DataLoader(dataset_train, batch_size=2, collate_fn=Semantic3dObjectReferanceDataset.collate_fn)

    for epoch in range(8):
        batch = next(iter(dataloader_train))
        optimizer.zero_grad()

        inputs = torch.tensor(batch['objects_positions'], dtype=torch.float)
        # targets = torch.tensor(batch['offset_vectors'], dtype=torch.float)

        out = model(inputs)
        loss = criterion.forward(out, targets)
        
        loss.backward()
        optimizer.step()

        print(f'\r epoch {epoch} loss {loss.item()}')
        # dict_losses[lr].append(loss.item())
    print('\n\n')

# plot_metrics({'loss': dict_losses}, None, show_plot=True, size=(8,5))