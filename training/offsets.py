import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torch_geometric.transforms as T 

import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict

from models.offset_regression import OffsetRegressor

from datapreparation.kitti360.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_TEST
from dataloading.kitti360.poses import Kitti360FineDatasetMulti, Kitti360FineDataset

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import calc_pose_error

def train_epoch(model, dataloader, args):
    model.train()
    losses = []

    for i_batch, batch in enumerate(dataloader):
        optimizer.zero_grad()
        preds = model(batch['hint_descriptions'])
        loss = criterion(preds, torch.tensor(batch['offsets'], dtype=torch.float, device=DEVICE))

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    
    return np.mean(losses)

@torch.no_grad()
def eval_epoch(model, dataloader, args):
    stats = EasyDict(
        pose_mid = [],
        pose_mean = [],
        pose_offsets = []
    )

    for i_batch, batch in enumerate(dataloader):
        preds = model(batch['hint_descriptions'])
        preds = preds.detach().cpu().numpy()

        # Calculations with ground-truth matches!
        # TODO: is it correct with the matches like this?
        stats.pose_mid.append(      calc_pose_error(batch['objects'], batch['matches'], batch['poses'], preds, use_mid_pred=True))
        stats.pose_mean.append(     calc_pose_error(batch['objects'], batch['matches'], batch['poses'], np.zeros_like(preds)))
        stats.pose_offsets.append(  calc_pose_error(batch['objects'], batch['matches'], batch['poses'], preds))

    for key in stats.keys():
        stats[key] = np.mean(stats[key])
    return stats

if __name__ == "__main__":
    args = parse_arguments()

    dataset_name = args.base_path[:-1] if args.base_path.endswith('/') else args.base_path
    dataset_name = dataset_name.split('/')[-1]
    print(f'Directory: {dataset_name}')

    plot_path = f'./plots/{dataset_name}/Offsets_bs{args.batch_size}_e{args.regressor_dim}_l-{args.regressor_learn}_ev{args.regressor_eval}_s{args.shuffle}_g{args.lr_gamma}'
    print('Plot:', plot_path, '\n')

    # Load data sets
    # ['2013_05_28_drive_0003_sync', ]
    transform = T.Compose([T.FixedPoints(32), ])
    dataset_train = Kitti360FineDatasetMulti(args.base_path, ['2013_05_28_drive_0003_sync', ], transform, args, flip_pose=False)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=Kitti360FineDataset.collate_fn, shuffle=args.shuffle)

    dataset_val = Kitti360FineDatasetMulti(args.base_path, ['2013_05_28_drive_0003_sync', ], transform, args, flip_pose=False)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=Kitti360FineDataset.collate_fn, shuffle=False)

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', DEVICE, torch.cuda.get_device_name(0))
    torch.autograd.set_detect_anomaly(True)    

    # Start training
    learning_rates = np.logspace(-2, -4, 3)

    train_stats_loss = {lr: [] for lr in learning_rates}
    train_stats_pose_mid = {lr: [] for lr in learning_rates}
    train_stats_pose_mean = {lr: [] for lr in learning_rates}
    train_stats_pose_offsets = {lr: [] for lr in learning_rates}

    val_stats_pose_mid = {lr: [] for lr in learning_rates}
    val_stats_pose_mean = {lr: [] for lr in learning_rates}
    val_stats_pose_offsets = {lr: [] for lr in learning_rates}

    best_val_offsets = -1

    for lr in learning_rates:
        model = OffsetRegressor(dataset_train.get_known_words(), args)
        model.to(DEVICE)

        criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,args.lr_gamma)    

        for epoch in range(args.epochs):
            loss = train_epoch(model, dataloader_train, args)
            train_stats_loss[lr].append(loss)

            stats_train = eval_epoch(model, dataloader_train, args)
            train_stats_pose_mid[lr].append(stats_train.pose_mid)
            train_stats_pose_mean[lr].append(stats_train.pose_mean)
            train_stats_pose_offsets[lr].append(stats_train.pose_offsets)

            stats_val = eval_epoch(model, dataloader_val, args)
            val_stats_pose_mid[lr].append(stats_val.pose_mid)
            val_stats_pose_mean[lr].append(stats_val.pose_mean)
            val_stats_pose_offsets[lr].append(stats_val.pose_offsets)            

            scheduler.step()

            print((
                f'\t epoch {epoch}, lr {lr:0.6},'
                f't-mid {stats_train.pose_mid}, t-mean {stats_train.pose_mean}, t-offsets {stats_train.pose_offsets},'
                f'v-mid {stats_val.pose_mid}, v-mean {stats_val.pose_mean}, v-offsets {stats_val.pose_offsets}'
            ), flush=True)


        if stats_val.pose_offsets > best_val_offsets:
            pass




    



    



    


