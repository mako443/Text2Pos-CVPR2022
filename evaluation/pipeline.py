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
from datapreparation.kitti360.utils import SCENE_NAMES as SCENE_NAMES_K360, SCENE_NAMES_TRAIN as SCENE_NAMES_TRAIN_K360, SCENE_NAMES_TEST as SCENE_NAMES_TEST_K360
from training.cell_retrieval2 import eval_epoch as eval_epoch_retrieval

'''
TODO:
- CARE: same location across different scenes!
- get_pos_in_cell() in SuperGlueMatch class
- How to handle orientation predictions?
'''

def run_retrieval(model, dataloader):
    accuracies, retrievals = eval_epoch_retrieval(model, dataloader, EasyDict(ranking_loss='pairwise', top_k=args.top_k))
    retrievals = [retrievals[idx] for idx in range(len(retrievals))] # Dict -> list
    print('Accs:')
    print(accuracies)
    return retrievals

if __name__ == '__main__':
    args = parse_arguments()
    print(args, "\n")
    threshs = [60, ] # List of list therfore not in arguments

    # Load datasets
    dataset_retrieval = Kitti360CellDatasetMulti('./data/kitti360', SCENE_NAMES_TEST_K360)
    dataset_matching = Kitti360PoseReferenceDatasetMulti('./data/kitti360', SCENE_NAMES_TEST_K360, args, split=None)
    assert len(dataset_retrieval) == len(dataset_matching) # If poses and cells become separate, it will need dedicated handling
    dataloader_retrieval = DataLoader(dataset_retrieval, batch_size=args.batch_size, collate_fn=Kitti360CellDataset.collate_fn)
    dataloader_matching = DataLoader(dataset_matching, batch_size=args.batch_size, collate_fn=Kitti360PoseReferenceDataset.collate_fn)

    # Load retrieval model
    model_retrieval = torch.load(args.path_retrieval)

    # Load matching model

    # Run retrieval
    retrievals = run_retrieval(model_retrieval, dataloader_retrieval)
    pos_in_cell = [np.array((0.5, 0.5)) for i in range(len(dataset_retrieval))] # Estimate middle of the cell for each retrieval
    accuracies = eval_pose_accuracies(dataset_retrieval, retrievals, pos_in_cell, top_k=args.top_k, threshs=threshs)
    print_accuracies(accuracies)

    # Run matching