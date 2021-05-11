import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torch_geometric.transforms as T 

import time
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict

from models.superglue_matcher import SuperGlueMatch
from models.tf_matcher import TransformerMatch

from dataloading.semantic3d.semantic3d import Semantic3dPoseReferenceMockDataset, Semantic3dPoseReferenceDataset, Semantic3dPoseReferenceDatasetMulti
from dataloading.kitti360.poses import Kitti360PoseReferenceDataset, Kitti360PoseReferenceDatasetMulti, Kitti360PoseReferenceMockDataset
from dataloading.kitti360.synthetic import Kitti360PoseReferenceMockDatasetPoints

from datapreparation.semantic3d.imports import COLORS as COLORS_S3D, COLOR_NAMES as COLOR_NAMES_S3D
from datapreparation.kitti360.utils import COLORS as COLORS_K360, COLOR_NAMES as COLOR_NAMES_K360
from datapreparation.kitti360.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_TEST

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, calc_recall_precision, calc_pose_error

'''
TODO:
- Aux. train color + class helpful?
- Which augmentation: RandomFlips, RandomRotate, Nothing? -> Not much difference?
- 512 points ok? -> 1024 maye slightly better but seems ok. Possibly re-check w/ aux-loss
- Merge differently / variations?
- Pre-train helpful?
- Re-formulate forward() as in CellRetrieval regarding features/embedding

- Refactoring: train (on-top, classes, center/closest point, color rgb/text, )
- feature ablation
- regress offsets: is error more in direction or magnitude? optimize?
- Pad at (0.5,0.5) for less harmfull miss-matches?
- Variable num_mentioned? (Care also at other places, potentially use lists there)

NOTES:
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

        loss.backward()
        optimizer.step()

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
    offset_vectors = []
    matches0_vectors = []

    for i_batch, batch in enumerate(dataloader):
        output = model(batch['objects'], batch['hint_descriptions'], batch['object_points'])
        offset_vectors.append(output.offsets.detach().cpu().numpy())
        matches0_vectors.append(output.matches0.detach().cpu().numpy())

        recall, precision = calc_recall_precision(batch['matches'], output.matches0.cpu().detach().numpy(), output.matches1.cpu().detach().numpy())
        stats.recall.append(recall)
        stats.precision.append(precision)

        stats.pose_mid.append(calc_pose_error(batch['objects'], output.matches0.detach().cpu().numpy(), batch['poses'], offsets=output.offsets.detach().cpu().numpy(), use_mid_pred=True))
        stats.pose_mean.append(calc_pose_error(batch['objects'], output.matches0.detach().cpu().numpy(), batch['poses'], offsets=None))
        stats.pose_offsets.append(calc_pose_error(batch['objects'], output.matches0.detach().cpu().numpy(), batch['poses'], offsets=output.offsets.detach().cpu().numpy()))        

    for key in stats.keys():
        stats[key] = np.mean(stats[key])
    return stats

if __name__ == "__main__":
    args = parse_arguments()
    print(args, "\n")

    '''
    Create data loaders
    '''    
    if args.dataset == 'S3D':
        scene_names = ('bildstein_station1_xyz_intensity_rgb','domfountain_station1_xyz_intensity_rgb','neugasse_station1_xyz_intensity_rgb','sg27_station1_intensity_rgb','sg27_station2_intensity_rgb','sg27_station4_intensity_rgb','sg27_station5_intensity_rgb','sg27_station9_intensity_rgb','sg28_station4_intensity_rgb','untermaederbrunnen_station1_xyz_intensity_rgb')
        dataset_val = Semantic3dPoseReferenceDatasetMulti('./data/numpy_merged/', './data/semantic3d', scene_names, args.pad_size, split=None)
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=Semantic3dPoseReferenceDataset.collate_fn)  
        
        dataset_train = Semantic3dPoseReferenceMockDataset(args, dataset_val.get_known_classes(), COLORS_S3D, COLOR_NAMES_S3D)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=Semantic3dPoseReferenceMockDataset.collate_fn)

    if args.dataset == 'K360':
        # train_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.RandomRotate(180, axis=2), T.NormalizeScale()])
        train_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.RandomRotate(120, axis=2), T.NormalizeScale()])
        dataset_train = Kitti360PoseReferenceMockDatasetPoints(args.base_path, SCENE_NAMES_TRAIN, train_transform, args, length=1024)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=Kitti360PoseReferenceMockDatasetPoints.collate_fn)

        print('CARE: Re-set scenes')
        val_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])
        dataset_val = Kitti360PoseReferenceDatasetMulti(args.base_path, SCENE_NAMES_TEST, val_transform, args, split=None)
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=Kitti360PoseReferenceDataset.collate_fn)  
        
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
    learning_rates = np.logspace(-3.0, -4.0 ,3) #[args.lr_idx : args.lr_idx + 1] # Larger than -3 throws error (even with warm-up)

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

            if scheduler: 
                scheduler.step()

            print((
                f'\t lr {lr:0.6} epoch {epoch} loss {train_out.loss:0.3f} '
                f't-recall {train_out.recall:0.2f} t-precision {train_out.precision:0.2f} t-mean {train_out.pose_mean:0.2f} t-offset {train_out.pose_offsets:0.2f} '
                f'v-recall {val_out.recall:0.2f} v-precision {val_out.precision:0.2f} v-mean {val_out.pose_mean:0.2f} v-offset {val_out.pose_offsets:0.2f} '
                ))
        print()

        acc = np.mean((val_out.recall, val_out.precision))
        if acc > best_val_recallPrecision:
            model_path = f"./checkpoints/fine_acc{acc:0.2f}_lr{args.lr_idx}_p{args.pointnet_numpoints}.pth"
            print('Saving model to', model_path)
            try:
                torch.save(model, model_path)
            except Exception as e:
                print('Error saving model!', str(e))
            best_val_recallPrecision = acc

    '''
    Save plots
    '''
    plot_name = f'Fine-Shift-9-Eval-{args.dataset}_bs{args.batch_size}_obj-{args.num_mentioned}-{args.pad_size}_e{args.embed_dim}_lr{args.lr_idx}_l{args.num_layers}_i{args.sinkhorn_iters}_v{args.variation}_p{args.pointnet_numpoints}_g{args.lr_gamma}.png'
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
        'val-pose_offsets': val_stats_pose_offsets,      
    }
    plot_metrics(metrics, './plots/'+plot_name)        

    