import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import time
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict

from models.superglue_matcher import SuperGlueMatch
# from models.graph_matcher import GraphMatch
from models.tf_matcher import TransformerMatch

from dataloading.semantic3d.semantic3d import Semantic3dPoseReferenceMockDataset, Semantic3dPoseReferenceDataset, Semantic3dPoseReferenceDatasetMulti
from dataloading.kitti360.poses import Kitti360PoseReferenceDataset, Kitti360PoseReferenceMockDataset

from datapreparation.semantic3d.imports import COLORS as COLORS_S3D, COLOR_NAMES as COLOR_NAMES_S3D
from datapreparation.kitti360.utils import COLORS as COLORS_K360, COLOR_NAMES as COLOR_NAMES_K360

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, calc_recall_precision, calc_pose_error

'''
TODO:
- Refactoring: train (on-top, classes, center/closest point, color rgb/text, )
- CARE / TODO: on-top in synthetic data!
- feature ablation
- regress offsets: is error more in direction or magnitude? optimize?
- Pad at (0.5,0.5) for less harmfull miss-matches?
- Variable num_mentioned?

NOTES:
- Random number of pads/distractors: acc. improved ✓
'''


def train_epoch(model, dataloader, args):
    model.train()
    epoch_losses = []
    epoch_recalls = []
    epoch_precisions = []
    epoch_pose_mid = []
    epoch_pose_mean = []
    epoch_pose_offsets = []
    t0 = time.time()
    printed=False
    for i_batch, batch in enumerate(dataloader):
        if args.max_batches is not None and i_batch >= args.max_batches:
            break

        optimizer.zero_grad()
        output = model(batch['objects'], batch['hint_descriptions'])

        loss_matching = criterion_matching(output.P, batch['all_matches'])
        loss_offsets = criterion_offsets(output.offsets, torch.tensor(batch['offsets'], dtype=torch.float, device=DEVICE))
        
        if not printed:
            print(f'{loss_matching.item():0.2f} - {loss_offsets.item():0.2f}')
            printed = True

        loss = loss_matching + loss_offsets # TODO/CARE: balance between? Currently on same magnitude (w/ and w/o norm, but cell ∈ [0,1])

        loss.backward()
        optimizer.step()

        recall, precision = calc_recall_precision(batch['matches'], output.matches0.cpu().detach().numpy(), output.matches1.cpu().detach().numpy())

        # TODO: batch_data = {x:x, y:y...}, for key in batch_data: epoch_data[key].append(batch_data[key])
        epoch_losses.append(loss.item())
        epoch_recalls.append(recall)
        epoch_precisions.append(precision)

        epoch_pose_mid.append(calc_pose_error(batch['objects'], output.matches0.detach().cpu().numpy(), batch['poses'], args, offsets=output.offsets.detach().cpu().numpy(), use_mid_pred=True))
        epoch_pose_mean.append(calc_pose_error(batch['objects'], output.matches0.detach().cpu().numpy(), batch['poses'], args, offsets=None))
        epoch_pose_offsets.append(calc_pose_error(batch['objects'], output.matches0.detach().cpu().numpy(), batch['poses'], args, offsets=output.offsets.detach().cpu().numpy()))

    # return np.mean(epoch_losses), np.mean(epoch_recalls), np.mean(epoch_precisions), time.time()-t0
    return EasyDict(
        loss=np.mean(epoch_losses), 
        recall=np.mean(epoch_recalls), 
        precision=np.mean(epoch_precisions), 
        pose_mid=np.mean(epoch_pose_mid), 
        pose_mean=np.mean(epoch_pose_mean), 
        pose_offsets=np.mean(epoch_pose_offsets), 
        time=time.time()-t0
    )

@torch.no_grad()
def val_epoch(model, dataloader, args):
    # model.eval() #TODO/CARE: set eval() or not?
    epoch_recalls = []
    epoch_precisions = []
    epoch_pose_mid = []
    epoch_pose_mean = []
    epoch_pose_offsets = []    
    for i_batch, batch in enumerate(dataloader):
        # color_input = batch['objects_colors'] if args.use_color else None
        # output = model(batch['objects_classes'], batch['objects_positions'], batch['hint_descriptions'], object_colors=color_input)
        output = model(batch['objects'], batch['hint_descriptions'])

        recall, precision = calc_recall_precision(batch['matches'], output.matches0.cpu().detach().numpy(), output.matches1.cpu().detach().numpy())
        epoch_recalls.append(recall)
        epoch_precisions.append(precision)

        epoch_pose_mid.append(calc_pose_error(batch['objects'], output.matches0.detach().cpu().numpy(), batch['poses'], args, offsets=output.offsets.detach().cpu().numpy(), use_mid_pred=True))
        epoch_pose_mean.append(calc_pose_error(batch['objects'], output.matches0.detach().cpu().numpy(), batch['poses'], args, offsets=None))
        epoch_pose_offsets.append(calc_pose_error(batch['objects'], output.matches0.detach().cpu().numpy(), batch['poses'], args, offsets=output.offsets.detach().cpu().numpy()))        

    # return np.mean(epoch_recalls), np.mean(epoch_precisions)
    return EasyDict(
        recall=np.mean(epoch_recalls), 
        precision=np.mean(epoch_precisions), 
        pose_mid=np.mean(epoch_pose_mid), 
        pose_mean=np.mean(epoch_pose_mean), 
        pose_offsets=np.mean(epoch_pose_offsets), 
    )    

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
        dataset_train = Kitti360PoseReferenceMockDataset(args)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=Kitti360PoseReferenceMockDataset.collate_fn)

        dataset_val = Kitti360PoseReferenceDataset('./data/kitti360', '2013_05_28_drive_0000_sync', args)
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=Kitti360PoseReferenceDataset.collate_fn)  
        
        # dataset_train = Semantic3dPoseReferenceMockDataset(args, dataset_val.get_known_classes(), COLORS_K360, COLOR_NAMES_K360)
        # dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=Semantic3dPoseReferenceMockDataset.collate_fn)

    # dataset_train = Semantic3dPoseReferenceMockDataset(args, dataset_val.get_known_classes(), length=1024)
    # dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=Semantic3dPoseReferenceMockDataset.collate_fn)

    print(sorted(dataset_train.get_known_classes()))
    print(sorted(dataset_val.get_known_classes()))
    print(sorted(dataset_train.get_known_words()))
    print(sorted(dataset_val.get_known_words()))
    train_words = dataset_train.get_known_words()
    for w in dataset_val.get_known_words():
        assert w in train_words
    assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes())        

    # TODO: turn back on for multi
    # assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes()) and sorted(dataset_train.get_known_words()) == sorted(dataset_val.get_known_words())
    
    data0 = dataset_train[0]
    batch = next(iter(dataloader_train))

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', DEVICE)
    torch.autograd.set_detect_anomaly(True)    

    '''
    Start training
    '''
    learning_rates = np.logspace(-2.5, -3.5 ,3) 
    dict_loss = {lr: [] for lr in learning_rates}
    dict_recall = {lr: [] for lr in learning_rates}
    dict_precision = {lr: [] for lr in learning_rates}
    dict_pose_mid = {lr: [] for lr in learning_rates}
    dict_pose_mean = {lr: [] for lr in learning_rates}
    dict_pose_offsets = {lr: [] for lr in learning_rates}
    dict_val_recall = {lr: [] for lr in learning_rates}
    dict_val_precision = {lr: [] for lr in learning_rates}    
    dict_val_pose_mid = {lr: [] for lr in learning_rates}
    dict_val_pose_mean = {lr: [] for lr in learning_rates}
    dict_val_pose_offsets = {lr: [] for lr in learning_rates}    
    
    for lr in learning_rates:
        model = SuperGlueMatch(dataset_train.get_known_classes(), dataset_train.get_known_words(), args)
        # model = GraphMatch(dataset_train.get_known_classes(), dataset_train.get_known_words(), args.embed_dim, args.k, args.sinkhorn_iters, args.num_layers, args.use_features)
        # model = TransformerMatch(dataset_train.get_known_classes(), dataset_train.get_known_words(), args)
        model.to(DEVICE)

        criterion_matching = MatchingLoss()
        criterion_offsets = nn.MSELoss()

        # Warm-up 
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,args.lr_gamma)

        for epoch in range(args.epochs):
            if epoch==3:
                optimizer = optim.Adam(model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer,args.lr_gamma)            

            # loss, train_recall, train_precision, epoch_time = train_epoch(model, dataloader_train, args)
            train_out = train_epoch(model, dataloader_train, args)
            dict_loss[lr].append(train_out.loss)
            dict_recall[lr].append(train_out.recall)
            dict_precision[lr].append(train_out.precision)
            dict_pose_mid[lr].append(train_out.pose_mid)
            dict_pose_mean[lr].append(train_out.pose_mean)
            dict_pose_offsets[lr].append(train_out.pose_offsets)

            # val_recall, val_precision = val_epoch(model, dataloader_val, args) #CARE: which loader for val!
            val_out = val_epoch(model, dataloader_val, args) #CARE: which loader for val!
            dict_val_recall[lr].append(val_out.recall)
            dict_val_precision[lr].append(val_out.precision)     
            dict_val_pose_mid[lr].append(val_out.pose_mid)
            dict_val_pose_mean[lr].append(val_out.pose_mean)
            dict_val_pose_offsets[lr].append(val_out.pose_offsets)                   

            if scheduler: 
                scheduler.step()

            print((
                f'\t lr {lr:0.6} epoch {epoch} loss {train_out.loss:0.3f} '
                f't-recall {train_out.recall:0.2f} t-precision {train_out.precision:0.2f} t-mean {train_out.pose_mean:0.2f} t-offset {train_out.pose_offsets:0.2f} '
                f'v-recall {val_out.recall:0.2f} v-precision {val_out.precision:0.2f} v-mean {val_out.pose_mean:0.2f} v-offset {val_out.pose_offsets:0.2f} '
                ))
        print()

    '''
    Save plots
    '''
    # plot_name = f'matching_bs{args.batch_size}_mb{args.max_batches}_dist{args.num_distractors}_e{args.embed_dim}_l{args.num_layers}_i{args.sinkhorn_iters}_c{args.use_color}_g{args.lr_gamma}.png'
    # plot_name = f'G-match_bs{args.batch_size}_mb{args.max_batches}_dist{args.num_distractors}_e{args.embed_dim}_l{args.num_layers}_i{args.sinkhorn_iters}_k{args.k}_f{"-".join(args.use_features)}_g{args.lr_gamma}.png'
    # plot_name = f'TF-match_bs{args.batch_size}_mb{args.max_batches}_dist{args.num_distractors}_e{args.embed_dim}_i{args.sinkhorn_iters}_f{"-".join(args.use_features)}_g{args.lr_gamma}.png'
    # plot_name = f'SG-match_bs{args.batch_size}_mb{args.max_batches}_obj-{args.num_mentioned}-{args.num_distractors}_e{args.embed_dim}_l{args.num_layers}_i{args.sinkhorn_iters}_f{"-".join(args.use_features)}_g{args.lr_gamma}.png'
    # plot_name = f'SG-PosePad_bs{args.batch_size}_mb{args.max_batches}_obj-{args.num_mentioned}-{args.num_distractors}_e{args.embed_dim}_l{args.num_layers}_i{args.sinkhorn_iters}_f{"-".join(args.use_features)}_g{args.lr_gamma}.png'
    plot_name = f'SG-Off-{args.dataset}_bs{args.batch_size}_mb{args.max_batches}_obj-{args.num_mentioned}-{args.pad_size}_e{args.embed_dim}_l{args.num_layers}_i{args.sinkhorn_iters}_f{"-".join(args.use_features)}_g{args.lr_gamma}.png'
    metrics = {
        'train-loss': dict_loss,
        'train-loss1': dict_loss,
        'train-loss2': dict_loss,
        'train-loss3': dict_loss,
        'train-recall': dict_recall,
        'train-precision': dict_precision,
        'train-pose-mean': dict_pose_mean,
        'train-pose-offsets': dict_pose_offsets,
        'val-recall': dict_val_recall,
        'val-precision': dict_val_precision,        
        'val-pose-mean': dict_val_pose_mean,
        'val-pose-offsets': dict_val_pose_offsets,        
    }
    plot_metrics(metrics, './plots/'+plot_name)        

    