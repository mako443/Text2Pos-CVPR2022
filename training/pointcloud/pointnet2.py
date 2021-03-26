import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader

import time
import numpy as np
import matplotlib.pyplot as plt

from models.pointcloud.pointnet2 import PointNet2
from dataloading.semantic3d_pointcloud import Semantic3dObjectDataset

from training.args import parse_arguments
from training.plots import plot_metrics

'''
TODO:
- compare to PyG example -> Not better...
- check dropout, PyG-MLP
- BN before ReLU?
- optimize ratio and radius (random search!)
- why shuffle bad?
'''


def train_epoch(model, dataloader, args):
    model.train()
    epoch_losses = []
    epoch_accs = []
    
    for i_batch, batch in enumerate(dataloader):
        if args.max_batches is not None and i_batch >= args.max_batches:
            break

        optimizer.zero_grad()
        pred = model(batch)

        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        acc = torch.sum(torch.argmax(pred, dim=-1) == batch.y).item() / len(pred)
        epoch_accs.append(acc)
    
    pred = torch.argmax(pred, dim=-1).cpu().numpy()
    # print(pred, batch.y.cpu().numpy())
    
    return np.mean(epoch_losses), np.mean(epoch_accs)

@torch.no_grad()
def val_epoch(model, dataloader, args):
    # model.eval() #TODO: yes/no?
    epoch_accs = []

    for i_batch, batch in enumerate(dataloader):
        pred = model(batch)
        acc = torch.sum(torch.argmax(pred, dim=-1) == batch.y).item() / len(pred)
        epoch_accs.append(acc)

    return np.mean(epoch_accs)


if __name__ == "__main__":
    args = parse_arguments()
    print(args, "\n")

    '''
    Create data loaders
    '''    
    dataset_train = Semantic3dObjectDataset('./data/numpy_merged/', './data/semantic3d', split='train')
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=False)
    dataset_val = Semantic3dObjectDataset('./data/numpy_merged/', './data/semantic3d', split='test')
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=False)    

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)
    torch.autograd.set_detect_anomaly(True)        

    '''
    Start training
    '''
    learning_reates = np.logspace(-3.0, -4.0, 3)
    dict_loss = {lr: [] for lr in learning_reates}    
    dict_acc = {lr: [] for lr in learning_reates}
    dict_acc_val = {lr: [] for lr in learning_reates}

    for lr in learning_reates:
        model = PointNet2(num_classes=len(dataset_train.known_classes))
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,args.lr_gamma)

        for epoch in range(args.epochs):
            loss, acc_train = train_epoch(model, dataloader_train, args)
            acc_val = val_epoch(model, dataloader_val, args)

            dict_loss[lr].append(loss)
            dict_acc[lr].append(acc_train)
            dict_acc_val[lr].append(acc_val)

            scheduler.step()

            print(f'\t lr {lr:0.6f} epoch {epoch} loss {loss: 0.3f} acc-train {acc_train: 0.2f} acc-val {acc_val:0.2f}')
        print()

    '''
    Save plots
    '''
    plot_name = f'PN2_bs{args.batch_size}_mb{args.max_batches}_s{args.shuffle}.png'
    metrics = {
        'train-loss': dict_loss,
        'train-acc': dict_acc,
        'val-acc': dict_acc_val      
    }
    plot_metrics(metrics, './plots/'+plot_name)        