import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import time
import numpy as np
import matplotlib.pyplot as plt

from models.superglue_matcher import SuperGlueMatch
# from models.graph_matcher import GraphMatch
from models.tf_matcher import TransformerMatch
from dataloading.semantic3d import Semantic3dPoseReferanceMockDataset, Semantic3dPoseReferanceDataset

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, calc_recall_precision

'''
TODO:
- if not working: mock-train texts to texts (english - german)
- regress offsets: classify, discretized vector, actual vector
'''

def train_epoch(model, dataloader, args):
    model.train()
    epoch_losses = []
    epoch_recalls = []
    epoch_precisions = []
    t0 = time.time()
    for i_batch, batch in enumerate(dataloader):
        if args.max_batches is not None and i_batch >= args.max_batches:
            break

        optimizer.zero_grad()
        # color_input = batch['objects_colors'] if args.use_color else None
        # output = model(batch['objects_classes'], batch['objects_positions'], batch['hint_descriptions'], object_colors=color_input)
        output = model(batch['objects'], batch['hint_descriptions'])

        loss = criterion(output.P, batch['all_matches'])
        # print(f'\t\t batch {i_batch} loss {loss.item(): 0.3f}')

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        recall, precision = calc_recall_precision(batch['matches'], output.matches0.cpu().detach().numpy(), output.matches1.cpu().detach().numpy())
        epoch_recalls.append(recall)
        epoch_precisions.append(precision)

        # if i_batch == args.max_batches - 1:
        #     print('P')
        #     print(output.P.cpu().detach().numpy().astype(np.float16))

    return np.mean(epoch_losses), np.mean(epoch_recalls), np.mean(epoch_precisions), time.time()-t0

@torch.no_grad()
def val_epoch(model, dataloader, args):
    model.eval() #TODO/CARE: set eval() or not?
    epoch_recalls = []
    epoch_precisions = []
    for i_batch, batch in enumerate(dataloader):
        # color_input = batch['objects_colors'] if args.use_color else None
        # output = model(batch['objects_classes'], batch['objects_positions'], batch['hint_descriptions'], object_colors=color_input)
        output = model(batch['objects'], batch['hint_descriptions'])

        recall, precision = calc_recall_precision(batch['matches'], output.matches0.cpu().detach().numpy(), output.matches1.cpu().detach().numpy())
        epoch_recalls.append(recall)
        epoch_precisions.append(precision)

    return np.mean(epoch_recalls), np.mean(epoch_precisions)

if __name__ == "__main__":
    args = parse_arguments()
    print(args, "\n")
    
    '''
    Create data loaders
    '''    
    dataset_train = Semantic3dPoseReferanceMockDataset(6, pad_size=args.pad_size)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=Semantic3dPoseReferanceMockDataset.collate_fn)
    # dataset_val = Semantic3dObjectReferanceMockDataset(args.num_mentioned, args.num_distractors, length=256)
    # dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=Semantic3dObjectReferanceMockDataset.collate_fn)  
    dataset_val = Semantic3dPoseReferanceDataset('./data/numpy_merged/', './data/semantic3d', "bildstein_station1_xyz_intensity_rgb", pad_size=args.pad_size)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=Semantic3dPoseReferanceDataset.collate_fn)  

    assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes()) and sorted(dataset_train.get_known_words()) == sorted(dataset_val.get_known_words())
    
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
    dict_val_recall = {lr: [] for lr in learning_rates}
    dict_val_precision = {lr: [] for lr in learning_rates}    
    
    for lr in learning_rates:
        model = SuperGlueMatch(dataset_train.get_known_classes(), dataset_train.get_known_words(), args)
        # model = GraphMatch(dataset_train.get_known_classes(), dataset_train.get_known_words(), args.embed_dim, args.k, args.sinkhorn_iters, args.num_layers, args.use_features)
        # model = TransformerMatch(dataset_train.get_known_classes(), dataset_train.get_known_words(), args)
        model.to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = MatchingLoss()
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,args.lr_gamma)

        for epoch in range(args.epochs):
            loss, train_recall, train_precision, epoch_time = train_epoch(model, dataloader_train, args)
            dict_loss[lr].append(loss)
            dict_recall[lr].append(train_recall)
            dict_precision[lr].append(train_precision)

            val_recall, val_precision = val_epoch(model, dataloader_val, args) #CARE: which loader for val!
            dict_val_recall[lr].append(val_recall)
            dict_val_precision[lr].append(val_precision)            

            scheduler.step()

            print(f'\t lr {lr:0.6} epoch {epoch} loss {loss:0.3f} t-recall {train_recall:0.2f} t-precision {train_precision:0.2f} v-recall {val_recall:0.2f} v-precision {val_precision:0.2f} time {epoch_time:0.3f}')
        print()

    '''
    Save plots
    '''
    # plot_name = f'matching_bs{args.batch_size}_mb{args.max_batches}_dist{args.num_distractors}_e{args.embed_dim}_l{args.num_layers}_i{args.sinkhorn_iters}_c{args.use_color}_g{args.lr_gamma}.png'
    # plot_name = f'G-match_bs{args.batch_size}_mb{args.max_batches}_dist{args.num_distractors}_e{args.embed_dim}_l{args.num_layers}_i{args.sinkhorn_iters}_k{args.k}_f{"-".join(args.use_features)}_g{args.lr_gamma}.png'
    # plot_name = f'TF-match_bs{args.batch_size}_mb{args.max_batches}_dist{args.num_distractors}_e{args.embed_dim}_i{args.sinkhorn_iters}_f{"-".join(args.use_features)}_g{args.lr_gamma}.png'
    # plot_name = f'SG-match_bs{args.batch_size}_mb{args.max_batches}_obj-{args.num_mentioned}-{args.num_distractors}_e{args.embed_dim}_l{args.num_layers}_i{args.sinkhorn_iters}_f{"-".join(args.use_features)}_g{args.lr_gamma}.png'
    plot_name = f'SG-Pose_bs{args.batch_size}_mb{args.max_batches}_obj-{args.num_mentioned}-{args.pad_size}_e{args.embed_dim}_l{args.num_layers}_i{args.sinkhorn_iters}_f{"-".join(args.use_features)}_g{args.lr_gamma}.png'
    metrics = {
        'train-loss': dict_loss,
        'train-recall': dict_recall,
        'train-precision': dict_precision,
        'val-recall': dict_val_recall,
        'val-precision': dict_val_precision,        
    }
    plot_metrics(metrics, './plots/'+plot_name)        

    