import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torch_geometric.transforms as T 

import time
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict
import os
import os.path as osp

from models.superglue_matcher import SuperGlueMatch

from dataloading.kitti360.poses import Kitti360FineDataset, Kitti360FineDatasetMulti
# from dataloading.kitti360.synthetic import Kitti360FineSyntheticDataset

from datapreparation.semantic3d.imports import COLORS as COLORS_S3D, COLOR_NAMES as COLOR_NAMES_S3D
from datapreparation.kitti360.utils import COLORS as COLORS_K360, COLOR_NAMES as COLOR_NAMES_K360, SCENE_NAMES_TEST
from datapreparation.kitti360.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, calc_recall_precision, calc_pose_error

'''
RESULTS:
- train-offset always at 0.2, 0.15 before center-offset?
- val-offset and val-mean always at 0.2, 0.1 before center-offset?
- train recall + precision > 0.8
- val recall + precision 0.6 - 0.7
- best-cell: cell-shift is same, maybe even a little better
- pose-cell w/o shift: recall + precision slightly lower, accuracies ~ same
- pose-cell w/  shift: recall, precision and accs back at best+shift, 0.6-0.7
- See if improves on smaller threshold -> Yes!
- compare mean/offset accuracies with closest/center in offset-pred and pos_in_cell -> Not much difference in any case, all 0.11 with perfect matching

TODO:
- Try regress and match only direction (do not say "on-top", learn on centers)

- Handle or discuss objects gt-selection / overflow. 32 would be enough for most
- Merge differently / variations?

- feature ablation
- regress offsets: is error more in direction or magnitude? optimize?
- Pad at (0.5,0.5) for less harmfull miss-matches?

NOTES:
- Pre-train helpful? -> Apparently safer
- 512 points ok? -> 1024 maye slightly better but seems ok. Possibly re-check w/ aux-loss
- Prevent opposite-direction matches! -> Done with offset_closest comparison
- Random number of pads/distractors: acc. improved ✓
- Keep PN frozen? -> Bad
'''

def train_epoch(model, dataloader, args):
    model.train()
    stats = EasyDict(
        loss = [],
        loss_offsets = [],
        recall = [],
        precision = [],
        pose_mid = [],
        pose_mean = [],
        pose_offsets = [],
    )

    printed=False
    for i_batch, batch in enumerate(dataloader):
        if args.max_batches is not None and i_batch >= args.max_batches:
            break

        optimizer.zero_grad()
        output = model(batch['objects'], batch['hint_descriptions'], batch['object_points'])

        loss_matching = criterion_matching(output.P, batch['all_matches'])
        loss_offsets = criterion_offsets(output.offsets, torch.tensor(batch['offsets'], dtype=torch.float, device=DEVICE))
        # loss_classes = 0.5 * criterion_class(output.class_preds, torch.tensor(batch['object_class_indices'], dtype=torch.long, device=DEVICE).flatten())
        # loss_colors = 0.5 * criterion_color(output.color_preds, torch.tensor(batch['object_color_indices'], dtype=torch.long, device=DEVICE).flatten())
        
        loss = loss_matching + 5 * loss_offsets# + loss_classes + loss_colors # Currently fixed alpha seems enough, cell normed ∈ [0, 1]
        if not printed:
            print(f'Losses: {loss_matching.item():0.3f} {loss_offsets.item():0.3f}')
            printed = True

        try:
            loss.backward()
            optimizer.step()
        except Exception as e:
            print()
            print(str(e))
            print()
            print(batch['all_matches'])

        recall, precision = calc_recall_precision(batch['matches'], output.matches0.cpu().detach().numpy(), output.matches1.cpu().detach().numpy())

        stats.loss.append(loss.item())
        stats.loss_offsets.append(loss_offsets.item())
        stats.recall.append(recall)
        stats.precision.append(precision)

        stats.pose_mid.append(calc_pose_error(batch['objects'], output.matches0.detach().cpu().numpy(), batch['poses'], offsets=output.offsets.detach().cpu().numpy(), use_mid_pred=True))
        stats.pose_mean.append(calc_pose_error(batch['objects'], output.matches0.detach().cpu().numpy(), batch['poses'], offsets=None))
        stats.pose_offsets.append(calc_pose_error(batch['objects'], output.matches0.detach().cpu().numpy(), batch['poses'], offsets=output.offsets.detach().cpu().numpy()))

    for key in stats.keys():
        stats[key] = np.mean(stats[key])
    return stats

@torch.no_grad()
def eval_epoch(model, dataloader, args):
    # model.eval() #TODO/CARE: set eval() or not?

    stats = EasyDict(
        recall = [],
        precision = [],
        pose_mid = [],
        pose_mean = [],
        pose_offsets = [],
    )
    # offset_vectors = []
    # matches0_vectors = []

    for i_batch, batch in enumerate(dataloader):
        output = model(batch['objects'], batch['hint_descriptions'], batch['object_points'])
        # offset_vectors.append(output.offsets.detach().cpu().numpy())
        # matches0_vectors.append(output.matches0.detach().cpu().numpy())

        recall, precision = calc_recall_precision(batch['matches'], output.matches0.cpu().detach().numpy(), output.matches1.cpu().detach().numpy())
        stats.recall.append(recall)
        stats.precision.append(precision)

        stats.pose_mid.append(calc_pose_error(batch['objects'], output.matches0.detach().cpu().numpy(), batch['poses'], offsets=output.offsets.detach().cpu().numpy(), use_mid_pred=True))
        stats.pose_mean.append(calc_pose_error(batch['objects'], output.matches0.detach().cpu().numpy(), batch['poses'], offsets=None))
        stats.pose_offsets.append(calc_pose_error(batch['objects'], output.matches0.detach().cpu().numpy(), batch['poses'], offsets=output.offsets.detach().cpu().numpy()))        

    for key in stats.keys():
        stats[key] = np.mean(stats[key])
    return stats

def get_conf1(P):
    return np.sum(P[:, 0:-1, 0:-1])

def get_conf2(output):
    matches = output.matches0
    matching_scores = output.matching_scores0
    return matching_scores[matches>=0].sum().item()
    # if len(matching_scores) == 0:
    #     return 0.0
    # else:
    #     return matching_scores.sum().item()

@torch.no_grad()
def eval_conf(model, dataset, args):
    accs = []
    accs_old = []
    for i_sample in range(100):
        confs = []

        idx = np.random.randint(len(dataset))
        data = Kitti360FineDataset.collate_fn([dataset[idx], ])
        hints = data['hint_descriptions']
        output = model(data['objects'], hints, data['object_points'])
        
        matches = output.matches0.detach().cpu().numpy()
        confs.append(np.sum(matches >= 0))
         
        for _ in range(4):
            idx = np.random.randint(len(dataset))
            data = Kitti360FineDataset.collate_fn([dataset[idx], ])
            hints = data['hint_descriptions']
            output = model(data['objects'], hints, data['object_points'])
            matches = output.matches0.detach().cpu().numpy()
            confs.append(np.sum(matches >= 0))

        accs.append(np.argmax(confs) == 0)
        accs.append(np.argmax(np.flip(confs)) == len(confs)-1)
        accs_old.append(np.argmax(confs) == 0)

    print('Conf score:', np.mean(accs), np.mean(accs_old))   

if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(',','\n'), '\n')

    dataset_name = args.base_path[:-1] if args.base_path.endswith('/') else args.base_path
    dataset_name = dataset_name.split('/')[-1]
    print(f'Directory: {dataset_name}')

    cont = 'Y' if bool(args.continue_path) else 'N'
    plot_path = f'./plots/{dataset_name}/Fine_cont{cont}-bs{args.batch_size}_obj-{args.num_mentioned}-{args.pad_size}_e{args.embed_dim}_lr{args.lr_idx}_l{args.num_layers}_i{args.sinkhorn_iters}_v{args.variation}_p{args.pointnet_numpoints}_s{args.shuffle}_g{args.lr_gamma}_npa{int(args.no_pc_augment)}_nca{int(args.no_cell_augment)}.png'
    print('Plot:', plot_path, '\n')

    '''
    Create data loaders
    '''    
    if args.dataset == 'K360':
        if args.no_pc_augment:
            train_transform = T.FixedPoints(args.pointnet_numpoints)
            val_transform = T.FixedPoints(args.pointnet_numpoints)
        else:
            train_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.RandomRotate(120, axis=2), T.NormalizeScale()])                                    
            val_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])
                    
        dataset_train = Kitti360FineDatasetMulti(args.base_path, SCENE_NAMES_TRAIN, train_transform, args, flip_pose=False) # No cell-augment for fine
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=Kitti360FineDataset.collate_fn, shuffle=args.shuffle)

        dataset_val = Kitti360FineDatasetMulti(args.base_path, SCENE_NAMES_VAL, val_transform, args)
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=Kitti360FineDataset.collate_fn)  

        
    # print(sorted(dataset_train.get_known_classes()))
    # print(sorted(dataset_val.get_known_classes()))
    print(sorted(dataset_train.get_known_words()))
    print(sorted(dataset_val.get_known_words()))
    # train_words = dataset_train.get_known_words()
    # for w in dataset_val.get_known_words():
    #     assert w in train_words
    assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes())        

    # TODO: turn back on for multi
    # assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes()) and sorted(dataset_train.get_known_words()) == sorted(dataset_val.get_known_words())
    
    data0 = dataset_train[0]
    batch = next(iter(dataloader_train))

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', DEVICE, torch.cuda.get_device_name(0))
    torch.autograd.set_detect_anomaly(True)    

    best_val_recallPrecision = -1 # Measured by mean of recall and precision

    '''
    Start training
    '''
    learning_rates = np.logspace(-3.0, -4.0 ,3)[args.lr_idx : args.lr_idx + 1] # Larger than -3 throws error (even with warm-up)

    train_stats_loss = {lr: [] for lr in learning_rates}
    train_stats_loss_offsets = {lr: [] for lr in learning_rates}
    train_stats_recall = {lr: [] for lr in learning_rates}
    train_stats_precision = {lr: [] for lr in learning_rates}
    train_stats_pose_mid = {lr: [] for lr in learning_rates}
    train_stats_pose_mean = {lr: [] for lr in learning_rates}
    train_stats_pose_offsets = {lr: [] for lr in learning_rates}
    
    val_stats_recall = {lr: [] for lr in learning_rates}
    val_stats_precision = {lr: [] for lr in learning_rates}
    val_stats_pose_mid = {lr: [] for lr in learning_rates}
    val_stats_pose_mean = {lr: [] for lr in learning_rates}
    val_stats_pose_offsets = {lr: [] for lr in learning_rates}
    
    for lr in learning_rates:
        if bool(args.continue_path):
            model = torch.load(args.continue_path)
        else:
            model = SuperGlueMatch(dataset_train.get_known_classes(), COLOR_NAMES_K360, dataset_train.get_known_words(), args)
        model.to(DEVICE)

        criterion_matching = MatchingLoss()
        criterion_offsets = nn.MSELoss()
        criterion_class = nn.CrossEntropyLoss()
        criterion_color = nn.CrossEntropyLoss()

        # Warm-up 
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,args.lr_gamma)

        for epoch in range(args.epochs):
            if epoch==3:
                optimizer = optim.Adam(model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer,args.lr_gamma)     

            # loss, train_recall, train_precision, epoch_time = train_epoch(model, dataloader_train, args)
            train_out = train_epoch(model, dataloader_train, args)

            train_stats_loss[lr].append(train_out.loss)
            train_stats_loss_offsets[lr].append(train_out.loss_offsets)
            train_stats_recall[lr].append(train_out.recall)
            train_stats_precision[lr].append(train_out.precision)
            train_stats_pose_mid[lr].append(train_out.pose_mid)
            train_stats_pose_mean[lr].append(train_out.pose_mean)
            train_stats_pose_offsets[lr].append(train_out.pose_offsets)

            val_out = eval_epoch(model, dataloader_val, args) #CARE: which loader for val!
            val_stats_recall[lr].append(val_out.recall)
            val_stats_precision[lr].append(val_out.precision)     
            val_stats_pose_mid[lr].append(val_out.pose_mid)
            val_stats_pose_mean[lr].append(val_out.pose_mean)
            val_stats_pose_offsets[lr].append(val_out.pose_offsets)   

            print()
            eval_conf(model, dataset_val, args)
            print()

            if scheduler: 
                scheduler.step()

            print((
                f'\t lr {lr:0.6} epoch {epoch} loss {train_out.loss:0.3f} '
                f't-recall {train_out.recall:0.2f} t-precision {train_out.precision:0.2f} t-mean {train_out.pose_mean:0.2f} t-offset {train_out.pose_offsets:0.2f} '
                f'v-recall {val_out.recall:0.2f} v-precision {val_out.precision:0.2f} v-mean {val_out.pose_mean:0.2f} v-offset {val_out.pose_offsets:0.2f} '
                ), flush=True)
        print()

        acc = np.mean((val_out.recall, val_out.precision))
        if acc > best_val_recallPrecision:
            model_path = f"./checkpoints/{dataset_name}/fine_cont{cont}_acc{acc:0.2f}_lr{args.lr_idx}_obj-{args.num_mentioned}-{args.pad_size}_p{args.pointnet_numpoints}_npa{int(args.no_pc_augment)}_nca{int(args.no_cell_augment)}.pth"
            if not osp.isdir(osp.dirname(model_path)):
                os.mkdir(osp.dirname(model_path))

            print('Saving model to', model_path)
            try:
                torch.save(model, model_path)
            except Exception as e:
                print('Error saving model!', str(e))
            best_val_recallPrecision = acc

    '''
    Save plots
    '''
    metrics = {
        'train-loss': train_stats_loss,
        'train-loss_offsets': train_stats_loss_offsets,
        'train-recall': train_stats_recall,
        'train-precision': train_stats_precision,
        'train-pose_mid': train_stats_pose_mid,
        'train-pose_mean': train_stats_pose_mean,
        'train-pose_offsets': train_stats_pose_offsets,
        'val-recall': val_stats_recall,
        'val-precision': val_stats_precision,
        'val-pose_mid': val_stats_pose_mid,
        'val-pose_mean': val_stats_pose_mean,
        'val-pose_offsets': val_stats_pose_offsets          
    }
    if not osp.isdir(osp.dirname(plot_path)):
        os.mkdir(osp.dirname(plot_path))
    plot_metrics(metrics, plot_path)        

    
 