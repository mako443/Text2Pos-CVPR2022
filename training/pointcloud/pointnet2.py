import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

import time
import numpy as np
import matplotlib.pyplot as plt

from models.pointcloud.pointnet2 import PointNet2
from dataloading.semantic3d.semantic3d_pointcloud import Semantic3dObjectDataset, Semantic3dObjectDatasetMulti
from dataloading.kitti360.objects import Kitti360ObjectsDataset, Kitti360ObjectsDatasetMulti
from datapreparation.kitti360.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_TEST
from datapreparation.kitti360.utils import COLOR_NAMES as COLOR_NAMES_K360

from training.args import parse_arguments
from training.plots import plot_metrics

'''
TODO:
- train w/ color-pred or not?
- why shuffle bad?


NOTES:
- more points not helpful, but might be if better sampling earlier in pipeline
- Only normalize along largest dim -> Already how NormalizeScale works ✓
'''

def train_epoch(model, dataloader, args):
    model.train()
    epoch_losses = []
    epoch_accs = []
    epoch_accs_color = []
    
    for i_batch, batch in enumerate(dataloader):
        if args.max_batches is not None and i_batch >= args.max_batches:
            break

        optimizer.zero_grad()
        output = model(batch)

        # loss = 1/2 * (criterion(output.class_pred, batch.y) + criterion(output.color_pred, batch.y_color))
        loss = criterion(output.class_pred, batch.y)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        acc = torch.sum(torch.argmax(output.class_pred, dim=-1) == batch.y).item() / len(output.class_pred)
        epoch_accs.append(acc)
        
        # acc_color = torch.sum(torch.argmax(output.color_pred, dim=-1) == batch.y_color).item() / len(output.color_pred)
        epoch_accs_color.append(-1)
        
    return np.mean(epoch_losses), np.mean(epoch_accs), np.mean(epoch_accs_color)

@torch.no_grad()
def val_epoch(model, dataloader, args):
    model.eval() #TODO: yes/no?
    epoch_accs = []
    epoch_accs_color = []    

    for i_batch, batch in enumerate(dataloader):
        output = model(batch)
        acc = torch.sum(torch.argmax(output.class_pred, dim=-1) == batch.y).item() / len(output.class_pred)
        epoch_accs.append(acc)

        # acc_color = torch.sum(torch.argmax(output.color_pred, dim=1) == batch.y_color).item() / len(output.color_pred)
        epoch_accs_color.append(-1)

    return np.mean(epoch_accs), np.mean(epoch_accs_color)


if __name__ == "__main__":
    args = parse_arguments()
    print(args, "\n")

    print('CARE: TRAINING ORACLE MODEL!')

    '''
    Create data loaders
    '''    
    transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale(), T.RandomFlip(0), T.RandomFlip(1), T.RandomFlip(2), T.NormalizeScale()])

    if args.dataset == 'S3D':
        scene_names = ['bildstein_station1_xyz_intensity_rgb','domfountain_station1_xyz_intensity_rgb','neugasse_station1_xyz_intensity_rgb','sg27_station1_intensity_rgb','sg27_station2_intensity_rgb','sg27_station4_intensity_rgb','sg27_station5_intensity_rgb','sg27_station9_intensity_rgb','sg28_station4_intensity_rgb','untermaederbrunnen_station1_xyz_intensity_rgb']
        # dataset_train = Semantic3dObjectDataset('./data/numpy_merged/', './data/semantic3d', split='train')
        dataset_train = Semantic3dObjectDatasetMulti('./data/numpy_merged/', './data/semantic3d', scene_names, split='train', transform=transform)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=False)
        # dataset_val = Semantic3dObjectDataset('./data/numpy_merged/', './data/semantic3d', split='test')
        dataset_val = Semantic3dObjectDatasetMulti('./data/numpy_merged/', './data/semantic3d', scene_names, split='test')
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=False)    

    if args.dataset == 'K360':     
        # train_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.RandomRotate(180, axis=2), T.NormalizeScale()]) # This proved best
        # train_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.RandomRotate(120, axis=2), T.NormalizeScale()])                                    
        train_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

        dataset_train = Kitti360ObjectsDatasetMulti(args.base_path, SCENE_NAMES, split=None, transform=train_transform)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=args.shuffle)
        
        val_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])
        dataset_val = Kitti360ObjectsDatasetMulti(args.base_path, SCENE_NAMES_TEST, split=None, transform=val_transform)
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes())

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)
    torch.autograd.set_detect_anomaly(True)        

    '''
    Start training
    '''
    learning_reates = np.logspace(-2, -4.0, 5)[args.lr_idx : args.lr_idx + 1]
    dict_loss = {lr: [] for lr in learning_reates}    
    dict_acc = {lr: [] for lr in learning_reates}
    dict_acc_color = {lr: [] for lr in learning_reates}
    dict_acc_val = {lr: [] for lr in learning_reates}
    dict_acc_val_color = {lr: [] for lr in learning_reates}

    best_val_accuracy = -1

    for lr in learning_reates:
        model = PointNet2(num_classes=len(dataset_train.class_to_index), num_colors=len(COLOR_NAMES_K360), args=args)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,args.lr_gamma)

        for epoch in range(args.epochs):
            loss, acc_train, acc_train_color = train_epoch(model, dataloader_train, args)
            acc_val, acc_val_color = val_epoch(model, dataloader_val, args)

            dict_loss[lr].append(loss)
            dict_acc[lr].append(acc_train)
            dict_acc_val[lr].append(acc_val)
            dict_acc_color[lr].append(acc_train_color)
            dict_acc_val_color[lr].append(acc_val_color)

            scheduler.step()

            print(f'\t lr {lr:0.6f} epoch {epoch} loss {loss:0.3f} acc-train {acc_train:0.2f} acc-val {acc_val:0.2f}')

        if acc_val > best_val_accuracy:
            model_path = f"./checkpoints/pointnet_perfect_acc{acc_val:0.2f}_lr{args.lr_idx}_p{args.pointnet_numpoints}.pth"    
            print(f'Saving model to {model_path}')
            torch.save(model.state_dict(), model_path)
            best_val_accuracy = acc_val

        print()

    '''
    Save plots
    '''
    # plot_name = f'PN2_len{len(dataset_train)}_bs{args.batch_size}_mb{args.max_batches}_l{args.num_layers}_v{args.variation}_s{args.shuffle}_g{args.lr_gamma}.png'
    plot_name = f'PN2-Shift-9-Perfect-{args.dataset}_bs{args.batch_size}_mb{args.max_batches}_lr{args.lr_idx}_pl{args.pointnet_layers}_pv{args.pointnet_variation}_t{args.pointnet_transform}_p{args.pointnet_numpoints}_s{args.shuffle}_g{args.lr_gamma}.png'
    metrics = {
        'train-loss': dict_loss,
        'train-acc': dict_acc,
        'train-acc-color': dict_acc_color,
        'val-acc': dict_acc_val,
        'val-acc-color': dict_acc_val_color
    }
    plot_metrics(metrics, './plots/'+plot_name)        