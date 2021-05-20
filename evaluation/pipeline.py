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
from evaluation.utils import calc_sample_accuracies, print_accuracies

from dataloading.kitti360.cells import Kitti360CoarseDataset, Kitti360CoarseDatasetMulti
# from dataloading.kitti360.poses import Kitti360PoseReferenceDatasetMulti, Kitti360PoseReferenceDataset
from datapreparation.kitti360.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_TEST
from training.coarse import eval_epoch as eval_epoch_retrieval
from models.superglue_matcher import get_pos_in_cell

import torch_geometric.transforms as T 

'''
TODO:
- confirm that increased accuracy @30m is "around corner"
- How to handle orientation predictions?
'''

def run_coarse(model, dataloader, top_k, threshs):
    """Returns retrievals as [(cell_indices_i), (cell_indices_i+1), ...] with i ∈ [0, len(poses)-1]

    Args:
        model ([type]): [description]
        dataloader ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Get top-cells
    retrieval_accuracies, retrievals = eval_epoch_retrieval(model, dataloader, EasyDict(ranking_loss='pairwise', top_k=args.top_k))
    retrievals = [retrievals[idx] for idx in range(len(retrievals))] # Dict -> list
    print('Retrieval Accs:')
    print(retrieval_accuracies)
    assert len(retrievals) == len(dataloader.dataset.all_poses)

    # Gather the accuracies for each sample
    accuracies = {k: {t: [] for t in threshs} for k in top_k}
    for i_sample in range(len(retrievals)):
        pose = dataloader.dataset.all_poses[i_sample]
        top_cells = [dataloader.dataset.all_cells[cell_id] for cell_id in retrievals[i_sample]]
        pos_in_cell = 0.5 * np.ones((len(top_cells), 2)) # Predict cell-centers
        accs = calc_sample_accuracies(pose.pose_w, top_cells, pos_in_cell, top_k, threshs)

        for k in top_k:
            for t in threshs:
                accuracies[k][t].append(accs[k][t])

    for k in top_k:
        for t in threshs:
            accuracies[k][t] = np.mean(accuracies[k][t])

    return retrievals, accuracies

@torch.no_grad()
def run_fine(model, dataloader):
    raise Exception("Not udpated yet!")
    offsets = []
    matches0 = []    
    for i_batch, batch in enumerate(dataloader):
        output = model(batch['objects'], batch['hint_descriptions'], batch['object_points'])
        offsets.append(output.offsets.detach().cpu().numpy())
        matches0.append(output.matches0.detach().cpu().numpy())
    return np.vstack((offsets)), np.vstack((matches0))        

'''
- Eval accuracies directly in run_matching(), rename run_coarse(), run_fine()
- Use TopK-Dataset, for now no DataLoader
'''

if __name__ == '__main__':
    args = parse_arguments()
    print(args, "\n")
    threshs = [15, 30] # Later: List of tuples, therefore not in arguments

    # Load datasets
    transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])
    # dataset_retrieval = Kitti360CellDatasetMulti(args.base_path, SCENE_NAMES_TEST, transform, split=None)
    # dataset_matching = Kitti360PoseReferenceDatasetMulti(args.base_path, SCENE_NAMES_TEST, transform, args, split=None)
    # assert len(dataset_retrieval) == len(dataset_matching) # If poses and cells become separate, this will need dedicated handling
    
    # dataloader_retrieval = DataLoader(dataset_retrieval, batch_size=args.batch_size, collate_fn=Kitti360CellDataset.collate_fn)
    # dataloader_matching = DataLoader(dataset_matching, batch_size=args.batch_size, collate_fn=Kitti360PoseReferenceDataset.collate_fn)

    dataset_retrieval = Kitti360CoarseDatasetMulti(args.base_path, ['2013_05_28_drive_0003_sync', ], transform, shuffle_hints=False, flip_poses=False)
    dataloader_retrieval = DataLoader(dataset_retrieval, batch_size = args.batch_size, collate_fn=Kitti360CoarseDataset.collate_fn)

    dataset_cell_only = dataset_retrieval.get_cell_dataset()

    # Load models
    model_retrieval = torch.load(args.path_retrieval)
    model_matching = torch.load(args.path_matching)

    # Run coarse
    retrievals, coarse_accuracies = run_coarse(model_retrieval, dataloader_retrieval, args.top_k, threshs)
    print('Coarse: ')
    print_accuracies(coarse_accuracies)

    quit()
    # OLD

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
    
