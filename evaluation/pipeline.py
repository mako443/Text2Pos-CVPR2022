import numpy as np
import os
import os.path as osp
import cv2
from easydict import EasyDict

import torch
from torch.utils.data import DataLoader

from training.plots import plot_metrics
from training.losses import calc_pose_error
from evaluation.pipeline import parse_arguments

from dataloading.kitti360.cells import Kitti360CellDatasetMulti
from dataloading.kitti360.poses import Kitti360PoseReferenceDatasetMulti
from datapreparation.kitti360.utils import SCENE_NAMES as SCENE_NAMES_K360, SCENE_NAMES_TRAIN as SCENE_NAMES_TRAIN_K360, SCENE_NAMES_TEST as SCENE_NAMES_TEST_K360
from training.cell_retrieval2 import eval_epoch as eval_epoch_retrieval

def run_retrieval(model, dataloader):
    _, retrievals = eval_epoch_retrieval(model, dataloader, EasyDict(ranking_loss='pairwise', top_k=[]))
    return retrievals

if __name__ == '__main__':
    args = parse_arguments()
    print(args, "\n")

    # ZIEL: eval_pose_accuracy cell-ret only @ cell-size: same results as cell-ret

    # Load datasets
    dataset_retrieval = Kitti360CellDatasetMulti('./data/kitti360', SCENE_NAMES_TEST_K360)
    dataset_matching = Kitti360PoseReferenceDatasetMulti('./data/kitti360', SCENE_NAMES_TEST_K360, args, split=None)
    assert len(dataset_retrieval) == len(dataset_matching) # If poses and cells become separate, it will need dedicated handling
    dataloader_retrieval = DataLoader(dataset_retrieval, batch_size=args.batch_size, collate_fn=dataset_retrieval.collate_fn)
    dataloader_matching = DataLoader(dataset_matching, batch_size=args.batch_size, collate_fn=dataset_matching.collate_fn)

    # Load retrieval model

    # Load matching model

    # Run retrieval

    # Run matching