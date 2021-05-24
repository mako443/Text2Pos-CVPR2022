import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torch_geometric.transforms as T 

import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict
import os
import os.path as osp

from models.offset_regression import OffsetRegressor

from datapreparation.kitti360.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_TEST
from dataloading.kitti360.poses import Kitti360FineDatasetMulti, Kitti360FineDataset

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import calc_pose_error

'''
TODO:
- Train + eval all 8 of pose/best, learn closest/center, eval closest/center
- If necessary, update get_pose_in_cell() call in eval.pipeline, too...
- Flip or not?
- Use in Fine model
'''

def train_epoch(model, dataloader, args):
    model.train()
    losses = []

    for i_batch, batch in enumerate(dataloader):
        optimizer.zero_grad()
        preds = model(batch['hint_descriptions'])

        valid = torch.tensor(batch['offsets_valid'], device=DEVICE)
        targets = torch.tensor(batch['offsets'], dtype=torch.float, device=DEVICE)
        loss = criterion(preds[valid], targets[valid])

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
        batch_size = len(batch['poses'])

        preds = model(batch['hint_descriptions'])
        preds = preds.detach().cpu().numpy()

        # Calculations with ground-truth matches!
        # Re-create the matching-output from SuperGlue with gt-matches
        pad_size = np.max([len(matches_sample) for matches_sample in batch['matches']])
        matches = np.ones((batch_size, pad_size), dtype=np.int32) * -1
        for i, matches_sample in enumerate(batch['matches']):
            for (obj_idx, hint_idx) in matches_sample:
                matches[i, obj_idx] = hint_idx

        if args.regressor_eval == 'closest':
            stats.pose_mid.append(      calc_pose_error(batch['objects'], matches, batch['poses'], preds, use_mid_pred=True, debug_use_closest=True))
            stats.pose_mean.append(     calc_pose_error(batch['objects'], matches, batch['poses'], np.zeros_like(preds), debug_use_closest=True))
            stats.pose_offsets.append(  calc_pose_error(batch['objects'], matches, batch['poses'], preds, debug_use_closest=True))
        else:
            stats.pose_mid.append(      calc_pose_error(batch['objects'], matches, batch['poses'], preds, use_mid_pred=True))
            stats.pose_mean.append(     calc_pose_error(batch['objects'], matches, batch['poses'], np.zeros_like(preds)))
            stats.pose_offsets.append(  calc_pose_error(batch['objects'], matches, batch['poses'], preds))            


    for key in stats.keys():
        stats[key] = np.mean(stats[key])
    return stats

if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(',','\n'), '\n')

    dataset_name = args.base_path[:-1] if args.base_path.endswith('/') else args.base_path
    dataset_name = dataset_name.split('/')[-1]
    print(f'Directory: {dataset_name}')

    plot_path = f'./plots/{dataset_name}/Offsets_bs{args.batch_size}_e{args.regressor_dim}_rc-{args.regressor_cell}_rl-{args.regressor_learn}_re-{args.regressor_eval}_s{args.shuffle}_g{args.lr_gamma}.png'
    print('Plot:', plot_path, '\n')

    print('XXX CARE WRONG DIRECTORY XXX')

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
    learning_rates = np.logspace(-1, -3, 3)

    train_stats_loss = {lr: [] for lr in learning_rates}
    train_stats_pose_mid = {lr: [] for lr in learning_rates}
    train_stats_pose_mean = {lr: [] for lr in learning_rates}
    train_stats_pose_offsets = {lr: [] for lr in learning_rates}

    val_stats_pose_mid = {lr: [] for lr in learning_rates}
    val_stats_pose_mean = {lr: [] for lr in learning_rates}
    val_stats_pose_offsets = {lr: [] for lr in learning_rates}

    best_val_offsets = np.inf

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
                f'\t epoch {epoch}, lr {lr:0.6}, loss {loss:0.3f}, '
                f't-mid {stats_train.pose_mid:0.2f}, t-mean {stats_train.pose_mean:0.2f}, t-offsets {stats_train.pose_offsets:0.2f}, '
                f'v-mid {stats_val.pose_mid:0.2f}, v-mean {stats_val.pose_mean:0.2f}, v-offsets {stats_val.pose_offsets:0.2f}'
            ), flush=True)


        if stats_val.pose_offsets < best_val_offsets: # CARE: lower is better here!
            model_path = f"./checkpoints/{dataset_name}/offsets_acc{stats_val.pose_offsets:0.2f}.pth"
            if not osp.isdir(osp.dirname(model_path)):
                os.mkdir(osp.dirname(model_path))
            print('Saving model to', model_path)
            try:
                torch.save(model, model_path)
            except Exception as e:
                print('Error saving model:', str(e))
            best_val_offsets = stats_val.pose_offsets

    '''
    Save plots
    '''
    metrics = {
        'train-loss': train_stats_loss,
        'train-pose-mid': train_stats_pose_mid,
        'train-pose-mean': train_stats_pose_mean,
        'train-pose-offsets': train_stats_pose_offsets,
        'val-pose-mid': val_stats_pose_mid,
        'val-pose-mean': val_stats_pose_mean,
        'val-pose-offsets': val_stats_pose_offsets,
    }        
    if not osp.isdir(osp.dirname(plot_path)):
        os.mkdir(osp.dirname(plot_path))
    plot_metrics(metrics, plot_path)
    




    



    



    


