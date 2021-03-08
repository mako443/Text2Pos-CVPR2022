import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import time
import numpy as np
import matplotlib.pyplot as plt

from models.cell_retrieval import CellRetrievalNetwork
from dataloading.semantic3d import Semantic3dCellRetrievalDataset, Semantic3dCellRetrievalDatasetMulti

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import calc_recall_precision, MatchingLoss, PairwiseRankingLoss, HardestRankingLoss

'''
TODO:
- top-1 acc? Features, feature combine, loss, graph architecture
- validation set?
- try larger dataset?
'''

def eval_accuracy(obj_encodings, text_encodings, top_k, scoring):
    '''
    Calculates text->object L2-based retrieval accuracy
    Assumes obj_encodings and text_encodings in same shapes and order
    '''
    assert scoring in ('L2', 'cosine')
    assert obj_encodings.shape == text_encodings.shape and len(obj_encodings.shape)==2
    accuracies = {k: [] for k in top_k}
    for i in range(len(text_encodings)):
        if scoring == 'L2':
            diffs = np.linalg.norm(obj_encodings[:] - text_encodings[i], axis=1)
            sorted_indices = np.argsort(diffs) #Sort low->high
        if scoring == 'cosine':
            scores = obj_encodings[:] @ text_encodings[i]
            sorted_indices = np.argsort(-1.0 * scores) #Sort high->low
        for k in top_k:
            accuracies[k].append(i in sorted_indices[0:k])
    for k in top_k:
        accuracies[k] = np.mean(accuracies[k])
    return accuracies

def train_epoch(model, dataloader, args):
    model.train()
    epoch_losses = []
    obj_encodings = np.zeros((0, model.embed_dim))
    text_encodings = np.zeros((0, model.embed_dim))

    for i_batch, batch in enumerate(dataloader):
        if args.max_batches is not None and i_batch >= args.max_batches:
            break

        optimizer.zero_grad()
        anchor = model.encode_text(batch['descriptions'])
        positive = model.encode_objects(batch['objects'])

        if args.ranking_loss == 'triplet':
            negative = model.encode_objects(batch['negative_objects'])
            loss = criterion(anchor, positive, negative)
        else:
            loss = criterion(anchor, positive)

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        obj_encodings = np.vstack((obj_encodings, positive.cpu().detach().numpy()))
        text_encodings = np.vstack((text_encodings, anchor.cpu().detach().numpy()))

    accuracies = eval_accuracy(obj_encodings, text_encodings, args.top_k, 'L2' if args.ranking_loss=='triplet' else 'cosine')

    return np.mean(epoch_losses), accuracies

@torch.no_grad()
def val_epoch(model, dataloader, args):
    model.eval()
    obj_encodings = np.zeros((0, model.embed_dim))
    text_encodings = np.zeros((0, model.embed_dim)) 
    for i_batch, batch in enumerate(dataloader):
        optimizer.zero_grad()
        anchor = model.encode_text(batch['descriptions'])
        positive = model.encode_objects(batch['objects'])

        obj_encodings = np.vstack((obj_encodings, positive.cpu().detach().numpy()))
        text_encodings = np.vstack((text_encodings, anchor.cpu().detach().numpy()))  

    accuracies = eval_accuracy(obj_encodings, text_encodings, args.top_k, 'L2' if args.ranking_loss=='triplet' else 'cosine')

    return accuracies

if __name__ == "__main__":
    args = parse_arguments()
    print(args, "\n")
    
    '''
    Create data loaders
    '''    
    scene_names = ['bildstein_station1_xyz_intensity_rgb', 'sg27_station5_intensity_rgb']
    dataset_train = Semantic3dCellRetrievalDatasetMulti('./data/numpy_merged/', './data/semantic3d', scene_names, args.use_features)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=Semantic3dCellRetrievalDataset.collate_fn, shuffle=args.shuffle)
    data0 = dataset_train[0]
    batch = next(iter(dataloader_train))
    print(data0['descriptions'])

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
        if args.ranking_loss == 'triplet':
            criterion = nn.TripletMarginLoss(margin=args.margin)
        if args.ranking_loss == 'pairwise':
            criterion = PairwiseRankingLoss(margin=args.margin)
        if args.ranking_loss == 'hardest':
            criterion = HardestRankingLoss(margin=args.margin)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,args.lr_gamma)

        for epoch in range(args.epochs):
            loss, train_acc = train_epoch(model, dataloader_train, args)
            dict_loss[lr].append(loss)

            # val_acc = val_epoch(model, dataloader_train, args)
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
    plot_name = f'cellRet_len{len(dataset_train)}_bs{args.batch_size}_mb{args.max_batches}_e{args.embed_dim}_l-{args.ranking_loss}_m{args.margin}_f{"-".join(args.use_features)}.png'
    train_accs = {f'train-acc-{k}': dict_acc[k] for k in args.top_k}
    metrics = {
        'train-loss': dict_loss,
        **train_accs 
    }
    plot_metrics(metrics, './plots/'+plot_name)  



