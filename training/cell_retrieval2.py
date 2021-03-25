import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import time
import numpy as np
import matplotlib.pyplot as plt

from models.cell_retrieval import CellRetrievalNetwork
from dataloading.semantic3d_poses import Semantic3dPosesDataset, Semantic3dPosesDatasetMulti
from datapreparation.imports import COMBINED_SCENE_NAMES

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, PairwiseRankingLoss, HardestRankingLoss

'''
TODO:
- what about same best cells?!
- max-dist for descriptions?
'''

def train_epoch(model, dataloader, args):
    model.train()
    epoch_losses = []   

    for i_batch, batch in enumerate(dataloader):
        if args.max_batches is not None and i_batch >= args.max_batches:
            break  

        batch_size = len(batch['texts'])
 
        optimizer.zero_grad()
        anchor = model.encode_text(batch['texts']) 
        cell_objects = [cell.objects for cell in batch['cells']]
        positive = model.encode_objects(cell_objects)

        if args.ranking_loss == 'triplet':
            negative_cell_objects = [cell.objects for cell in batch['negative_cells']]
            negative = model.encode_objects(negative_cell_objects)
            loss = criterion(anchor, positive, negative)
        else:
            loss = criterion(anchor, positive)

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    return np.mean(epoch_losses)

@torch.no_grad()
def eval_epoch(model, dataloader, args, targets='all'):
    """Top-k retrieval for each pose against all cells in the dataset.
    Model might not have seen cells in pairwise-ranking-loss training.

    Args:
        model : The model
        dataloader : The dataloader
        args : Global arguments
        targets: <all> or <poses> to use all available cells as targets or only those matching a pose
    """
    assert targets in ('all', 'poses')
    dataset = dataloader.dataset
    
    #Encode all the cells
    cell_encodings = np.zeros((0, model.embed_dim))
    for i in range(0, len(dataset.cells), args.batch_size):
        cells = dataset.cells[i : i+args.batch_size]
        cell_objects = [cell.objects for cell in cells]
        encodings = model.encode_objects(cell_objects)
        cell_encodings = np.vstack((cell_encodings, encodings.cpu().detach().numpy()))

    #Encode all the poses
    pose_encodings = np.zeros((0, model.embed_dim))
    correct_indices = []
    for i_batch, batch in enumerate(dataloader):
        if args.max_batches is not None and i_batch >= args.max_batches:
            break 

        encodings = model.encode_text(batch['texts'])
        pose_encodings = np.vstack((pose_encodings, encodings.cpu().detach().numpy()))
        correct_indices.extend(batch['cell_indices'])
    assert len(correct_indices) == len(pose_encodings)

    if targets == 'poses': # Remove all the cells that are not the target of a pose
        for idx in range(len(cell_encodings)):
            if idx not in correct_indices:
                cell_encodings[idx, :] = np.inf

    accuracies = {k: [] for k in args.top_k}
    for i in range(len(pose_encodings)):
        if args.ranking_loss == 'triplet':
            dists = np.linalg.norm(cell_encodings[:] - pose_encodings[i], axis=1)
            sorted_indices = np.argsort(dists) #Low->high
        else:
            scores = cell_encodings[:] @ pose_encodings[i]
            sorted_indices = np.argsort(-1.0 * scores) #Sort high->low
            
        for k in args.top_k:
            accuracies[k].append(correct_indices[i] in sorted_indices[0:k])
    
    for k in args.top_k:
        accuracies[k] = np.mean(accuracies[k])
    return accuracies 

if __name__ == "__main__":
    args = parse_arguments()
    print(args, "\n")
    
    '''
    Create data loaders
    '''    
    # dataset_train = Semantic3dPosesDatasetMulti('./data/numpy_merged/', './data/semantic3d', COMBINED_SCENE_NAMES, args.cell_size, args.cell_stride, split='train')
    # dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=Semantic3dPosesDataset.collate_fn, shuffle=args.shuffle)
    # dataset_val = Semantic3dPosesDatasetMulti('./data/numpy_merged/', './data/semantic3d', COMBINED_SCENE_NAMES, args.cell_size, args.cell_stride, split='test')
    # dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=Semantic3dPosesDataset.collate_fn, shuffle=False)

    # cell_sizes = np.random.randint(low=50, high=75, size=6)
    # cell_strides = [np.random.randint(cs//4, cs//2) for cs in cell_sizes]
    # cell_sizes_strides = list(zip(cell_sizes, cell_strides))
    cell_sizes_strides = [(50, 50//2), (50, 50*2//3), (60, 60//2), (60, 60*2//3), (70, 70//2), (70, 70*2//3)]
    # cell_sizes_strides = [(20, 20), (20, 20*2//3), (30, 30), (30, 30*2//3), (40, 40), (40, 40*2//3)]
    print(cell_sizes_strides)

    dict_loss = {css: [] for css in cell_sizes_strides}
    dict_acc = {k: {css: [] for css in cell_sizes_strides} for k in args.top_k}
    dict_acc_val = {k: {css: [] for css in cell_sizes_strides} for k in args.top_k} 

    for cell_size, cell_stride in cell_sizes_strides: 
        dataset_train = Semantic3dPosesDataset('./data/numpy_merged/', './data/semantic3d', "sg27_station1_intensity_rgb", cell_size, cell_stride, split='train')
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=Semantic3dPosesDataset.collate_fn, shuffle=args.shuffle)
        dataset_val = Semantic3dPosesDataset('./data/numpy_merged/', './data/semantic3d', "sg27_station1_intensity_rgb", cell_size, cell_stride, split='test')
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=Semantic3dPosesDataset.collate_fn, shuffle=False)
        print("\t\t",cell_size, cell_stride, dataset_val.gather_stats())

        data = dataset_train[0]        
        batch = next(iter(dataloader_train))

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('device:', device)
        torch.autograd.set_detect_anomaly(True)     

        # learning_rates = np.logspace(-1.5,-4,6)[1:-1]
        # dict_loss = {lr: [] for lr in learning_rates}
        # dict_acc = {k: {lr: [] for lr in learning_rates} for k in args.top_k}
        # dict_acc_val = {k: {lr: [] for lr in learning_rates} for k in args.top_k}    

        ACC_TARGET = 'poses'
        # for lr in learning_rates:
        for lr in (0.001,):
            model = CellRetrievalNetwork(dataset_train.get_known_classes(), dataset_train.get_known_words(), args.embed_dim, k=args.k, use_features=args.use_features)
            model.to(device)    

            optimizer = optim.Adam(model.parameters(), lr=lr)
            if args.ranking_loss == 'pairwise':
                criterion = PairwiseRankingLoss(margin=args.margin)
            if args.ranking_loss == 'hardest':
                criterion = HardestRankingLoss(margin=args.margin)
            if args.ranking_loss == 'triplet':
                criterion = nn.TripletMarginLoss(margin=args.margin)

            scheduler = optim.lr_scheduler.ExponentialLR(optimizer,args.lr_gamma)

            for epoch in range(args.epochs):
                loss = train_epoch(model, dataloader_train, args)
                train_acc = eval_epoch(model, dataloader_train, args, targets=ACC_TARGET)
                # val_acc = {k: -1 for k in args.top_k}
                val_acc = eval_epoch(model, dataloader_val, args, targets=ACC_TARGET)

                key = (cell_size, cell_stride)
                dict_loss[key].append(loss)
                for k in args.top_k:
                    dict_acc[k][key].append(train_acc[k])
                    dict_acc_val[k][key].append(val_acc[k])

                scheduler.step()
                print(f'\t lr {lr:0.4} loss {loss:0.2f} epoch {epoch} train-acc: ', end="")
                for k, v in train_acc.items():
                    print(f'{k}-{v:0.2f} ', end="")
                print('val-acc: ', end="")
                for k, v in val_acc.items():
                    print(f'{k}-{v:0.2f} ', end="")            
                print()        

    '''
    Save plots
    '''
    plot_name = f'0cells_len{len(dataset_train)}_bs{args.batch_size}_mb{args.max_batches}_e{args.embed_dim}_l-{args.ranking_loss}_m{args.margin}_c{int(args.cell_size)}-{int(args.cell_stride)}_f{"-".join(args.use_features)}_t-{ACC_TARGET}.png'
    train_accs = {f'train-acc-{k}': dict_acc[k] for k in args.top_k}
    val_accs = {f'val-acc-{k}': dict_acc_val[k] for k in args.top_k}
    metrics = {
        'train-loss': dict_loss,
        **train_accs,
        **val_accs
    }
    plot_metrics(metrics, './plots/'+plot_name)              