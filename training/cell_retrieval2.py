import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import time
import numpy as np
import matplotlib.pyplot as plt

from models.cell_retrieval import CellRetrievalNetwork
from dataloading.semantic3d_poses import Semantic3dPosesDataset
from datapreparation.imports import COMBINED_SCENE_NAMES

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, PairwiseRankingLoss, HardestRankingLoss

'''
TODO:
- what about same best cells?!
- cell size and stride
- max-dist for descriptions
- multi-scene
'''

def train_epoch(model, dataloader, args):
    model.train()
    epoch_losses = []   

    for i_batch, batch in enumerate(dataloader):
        if args.max_batches is not None and i_batch >= args.max_batches:
            break  
 
        optimizer.zero_grad()
        anchor = model.encode_text(batch['texts']) 
        cell_objects = [cell.objects for cell in batch['cells']]
        positive = model.encode_objects(cell_objects)

        print(anchor.shape)
        print(positive.shape)
        quit()

        loss = criterion(anchor, positive)

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    return np.mean(epoch_losses)

@torch.no_grad()
def eval_epoch(model, dataloader, args):
    """Top-k retrieval for each pose against all cells in the dataset.
    Model might not have seen cells in pairwise-ranking-loss training.

    Args:
        model : The model
        dataloader : The dataloader
        args : Global argumentds
    """
    assert args.ranking_loss == 'pairwise' #Else use other distances!
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
        encodings = model.encode_text(batch['texts'])
        pose_encodings = np.vstack((pose_encodings, encodings.cpu().detach().numpy()))
        correct_indices.extend(batch['cell_indices'])
    assert len(correct_indices) == len(pose_encodings)

    accuracies = {k: [] for k in args.top_k}
    for i in range(len(pose_encodings)):
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
    scene_names = COMBINED_SCENE_NAMES #['bildstein_station1_xyz_intensity_rgb', 'sg27_station5_intensity_rgb']
    dataset_train = Semantic3dPosesDataset('./data/numpy_merged/', './data/semantic3d')
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=Semantic3dPosesDataset.collate_fn, shuffle=args.shuffle)
    data = dataset_train[0]        
    batch = next(iter(dataloader_train))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)
    torch.autograd.set_detect_anomaly(True)     

    learning_reates = np.logspace(-1.5,-4,6)[:-2]
    dict_loss = {lr: [] for lr in learning_reates}
    dict_acc = {k: {lr: [] for lr in learning_reates} for k in args.top_k}
    dict_acc_val = {k: {lr: [] for lr in learning_reates} for k in args.top_k}    

    for lr in learning_reates:
        model = CellRetrievalNetwork(dataset_train.get_known_classes(), dataset_train.get_known_words(), args.embed_dim, k=args.k, use_features=args.use_features)
        model.to(device)    

        optimizer = optim.Adam(model.parameters(), lr=lr)
        assert args.ranking_loss == 'pairwise'
        if args.ranking_loss == 'pairwise':
            criterion = PairwiseRankingLoss(margin=args.margin)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,args.lr_gamma)

        for epoch in range(args.epochs):
            loss = train_epoch(model, dataloader_train, args)
            train_acc = eval_epoch(model, dataloader_train, args)

            dict_loss[lr].append(loss)
            for k in args.top_k:
                dict_acc[k][lr].append(train_acc[k])
                # dict_acc_val[k][lr].append(val_acc[k])

            scheduler.step()
            print(f'\t lr {lr:0.4} epoch {epoch} train-acc: ', end="")
            for k, v in train_acc.items():
                print(f'{k}-{v:0.2f} ', end="")
            print()        

    '''
    Save plots
    '''
    plot_name = f'cellFree_len{len(dataset_train)}_bs{args.batch_size}_mb{args.max_batches}_e{args.embed_dim}_l-{args.ranking_loss}_m{args.margin}_f{"-".join(args.use_features)}.png'
    train_accs = {f'train-acc-{k}': dict_acc[k] for k in args.top_k}
    val_accs = {f'val-acc-{k}': dict_acc_val[k] for k in args.top_k}
    metrics = {
        'train-loss': dict_loss,
        **train_accs,
        # **val_accs
    }
    plot_metrics(metrics, './plots/'+plot_name)              