import numpy as np
import os
import os.path as osp
import cv2
from easydict import EasyDict

import torch
from torch.utils.data import DataLoader

from training.plots import plot_metrics
from training.losses import calc_pose_error
from evaluation.args import parse_arguments
from evaluation.utils import eval_pose_accuracies, print_accuracies

from dataloading.kitti360.cells import Kitti360CellDatasetMulti, Kitti360CellDataset
from dataloading.kitti360.poses import Kitti360PoseReferenceDatasetMulti, Kitti360PoseReferenceDataset
from datapreparation.kitti360.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_TEST
from training.coarse import eval_epoch as eval_epoch_retrieval
from models.superglue_matcher import get_pos_in_cell

import torch_geometric.transforms as T 

'''
TODO:
- confirm that increased accuracy @30m is "around corner"
- How to handle orientation predictions?
- CARE: same location across different scenes!
'''

def run_retrieval(model, dataloader):
    accuracies, _, retrievals = eval_epoch_retrieval(model, dataloader, EasyDict(ranking_loss='pairwise', top_k=args.top_k))
    retrievals = [retrievals[idx] for idx in range(len(retrievals))] # Dict -> list
    print('Accs:')
    print(accuracies)
    return retrievals

@torch.no_grad()
def run_matching(model, dataloader):
    offsets = []
    matches0 = []    
    for i_batch, batch in enumerate(dataloader):
        output = model(batch['objects'], batch['hint_descriptions'], batch['object_points'])
        offsets.append(output.offsets.detach().cpu().numpy())
        matches0.append(output.matches0.detach().cpu().numpy())
    return np.vstack((offsets)), np.vstack((matches0))        

if __name__ == '__main__':
    args = parse_arguments()
    print(args, "\n")
    threshs = [15, 30] # Later: List of tuples, therefore not in arguments

    # Load datasets
    transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])
    dataset_retrieval = Kitti360CellDatasetMulti(args.base_path, SCENE_NAMES_TEST, transform, split=None)
    dataset_matching = Kitti360PoseReferenceDatasetMulti(args.base_path, SCENE_NAMES_TEST, transform, args, split=None)
    assert len(dataset_retrieval) == len(dataset_matching) # If poses and cells become separate, this will need dedicated handling
    
    dataloader_retrieval = DataLoader(dataset_retrieval, batch_size=args.batch_size, collate_fn=Kitti360CellDataset.collate_fn)
    dataloader_matching = DataLoader(dataset_matching, batch_size=args.batch_size, collate_fn=Kitti360PoseReferenceDataset.collate_fn)

    # Load models
    model_retrieval = torch.load(args.path_retrieval)
    model_matching = torch.load(args.path_matching)

    # Run retrieval
    retrievals = run_retrieval(model_retrieval, dataloader_retrieval)
    pos_in_cell = [np.array((0.5, 0.5)) for i in range(len(dataset_retrieval))] # Estimate middle of the cell for each retrieval
    accuracies = eval_pose_accuracies(dataset_retrieval, retrievals, pos_in_cell, top_k=args.top_k, threshs=threshs)
    print_accuracies(accuracies)

    # Run matching
    offsets, matches0 = run_matching(model_matching, dataloader_matching)

    # Without offsets
    pos_in_cell = [get_pos_in_cell(dataset_matching[i]['objects'], matches0[i], np.zeros_like(offsets[i])) for i in range(len(dataset_matching))] # Zero-offsets to just take mean of objects
    accuracies = eval_pose_accuracies(dataset_retrieval, retrievals, pos_in_cell, top_k=args.top_k, threshs=threshs)
    print_accuracies(accuracies)

    # With offsets
    pos_in_cell = [get_pos_in_cell(dataset_matching[i]['objects'], matches0[i], offsets[i]) for i in range(len(dataset_matching))] # Using actual offset-vectors
    accuracies = eval_pose_accuracies(dataset_retrieval, retrievals, pos_in_cell, top_k=args.top_k, threshs=threshs)    
    print_accuracies(accuracies)
    
