import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch_geometric.nn as gnn
from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

import numpy as np
import os
import os.path as osp
import pickle
import matplotlib.pyplot as plt

from models.modules import get_mlp, LanguageEncoder

path = './data/ModelNet10'
pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
train_dataset = ModelNet(path, '10', True, transform, pre_transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
test_dataset = ModelNet(path, '10', False, transform, pre_transform)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=1)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = gnn.DynamicEdgeConv(get_mlp([2 * 3, 64, 64]), k=20, aggr='max')
        self.conv2 = gnn.DynamicEdgeConv(get_mlp([2 * 64, 128]), k=20, aggr='max')

        self.mlp1 = get_mlp([128 + 64, 512])
        self.mlp2 = get_mlp([1024, 512, 256, 10])

    def forward(self, data):
        x, batch = data.pos, data.batch
        out1 = self.conv1(x)
        out2 = self.conv2(x)

        x = torch.cat((out1, out2), dim=1)
        x = self.mlp1(x)

        x = gnn.global_mean_pool(x, batch)

        x = self.mlp2(x)

        return x

def train_epoch(model, dataloader):
    model.train()
    epoch_losses = []
    epoch_accs = []

    for i_batch, batch in enumerate(dataloader):
        optimizer.zero_grad()

        out = model(batch)
        loss = criterion(out, batch.y)
        
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(out, dim=-1)
            acc = torch.mean(preds == batch.y)

        epoch_losses.append(loss.item())
        epoch_accs.append(acc.item())

    return np.mean(epoch_losses), np.mean(epoch_accs)

@torch.no_grad()
def val_epoch(model, dataloader):
    model.eval()
    epoch_accs = []

    for i_batch, batch in enumerate(dataloader):
        out = model(batch)
        preds = torch.argmax(out, dim=-1)
        acc = torch.mean(preds == batch.y)

        epoch_accs.append(acc.item())
        
    return np.mean(epoch_accs)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Net().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_losses = []
train_accs = []
val_accs = []

for epoch in range(32):
    loss, acc = train_epoch(model, train_loader)
    train_losses.append(loss)
    train_accs.append(acc)

    acc = val_epoch(model, test_loader)
    val_accs.append(acc)
    
    print(f'epoch {epoch}, loss {loss:0.3f} train-acc {train_accs[-1] : 0.3f}, val-acc {val_accs[-1] : 0.3f}')

plt.figure()
plt.subplot(2,2,1)
plt.plot(train_losses)
plt.title('Losses')
plt.gca().set_ylim(bottom=0.0) #Set the bottom to 0.0

plt.subplot(2,2,2)
plt.plot(train_accs)
plt.title('Train acc')
plt.gca().set_ylim(bottom=0.0) #Set the bottom to 0.0
plt.legend()

plt.subplot(2,2,4)
plt.plot(val_accs)
plt.title('Val acc')
plt.gca().set_ylim(bottom=0.0) #Set the bottom to 0.0
plt.legend()

