import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import time
import numpy as np
import matplotlib.pyplot as plt

from models.superglue_matcher import SuperGlueMatch
from dataloading.semantic3d import Semantic3dObjectReferanceDataset

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, calc_recall_precision

'''
TODO:
- if not working: mock-train texts to texts (english - german)
- try to encode object colors
- care: what about ambivalent matches?
- batched training good/bad?
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

    return np.mean(epoch_losses), np.mean(epoch_recalls), np.mean(epoch_precisions), time.time()-t0

@torch.no_grad()
def val_epoch(model, dataloader, args):
    model.eval() #TODO/CARE: set eval() or not?
    epoch_recalls = []
    epoch_precisions = []
    for i_batch, batch in enumerate(dataloader):
        #color_input = batch['objects_colors'] if args.use_color else None
        #output = model(batch['objects_classes'], batch['objects_positions'], batch['hint_descriptions'], object_colors=color_input)
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
    dataset_train = Semantic3dObjectReferanceDataset('./data/numpy_merged/', './data/semantic3d', num_distractors=args.num_distractors, split='train')
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=Semantic3dObjectReferanceDataset.collate_fn)
    data0 = dataset_train[0]
    batch = next(iter(dataloader_train))

    dataset_val = Semantic3dObjectReferanceDataset('./data/numpy_merged/', './data/semantic3d', num_distractors=args.num_distractors, split='test')
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=Semantic3dObjectReferanceDataset.collate_fn)

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', DEVICE)
    torch.autograd.set_detect_anomaly(True)    

    '''
    Start training
    '''
    learning_reates = np.logspace(-3,-5,5)[2:]
    dict_loss = {lr: [] for lr in learning_reates}
    dict_recall = {lr: [] for lr in learning_reates}
    dict_precision = {lr: [] for lr in learning_reates}
    dict_val_recall = {lr: [] for lr in learning_reates}
    dict_val_precision = {lr: [] for lr in learning_reates}    
    
    for lr in learning_reates:
        model = SuperGlueMatch(dataset_train.get_known_classes(), dataset_train.get_known_words(), args)
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
    plot_name = f'SG-Match_bs{args.batch_size}_mb{args.max_batches}_dist{args.num_distractors}_e{args.embed_dim}_l{args.num_layers}_i{args.sinkhorn_iters}_g{args.lr_gamma}.png'
    metrics = {
        'train-loss': dict_loss,
        'train-recall': dict_recall,
        'train-precision': dict_precision,
        'val-recall': dict_val_recall,
        'val-precision': dict_val_precision,        
    }
    plot_metrics(metrics, './plots/'+plot_name)        

    