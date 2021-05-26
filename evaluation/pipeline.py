import numpy as np
import os
import os.path as osp
import cv2
from easydict import EasyDict

import torch
from torch.utils.data import DataLoader
import time

# from training.losses import calc_pose_error # This cannot be used here anymore!!
from evaluation.args import parse_arguments
from evaluation.utils import calc_sample_accuracies, print_accuracies

from dataloading.kitti360.cells import Kitti360CoarseDataset, Kitti360CoarseDatasetMulti
from dataloading.kitti360.eval import Kitti360TopKDataset

from datapreparation.kitti360.utils import SCENE_NAMES_TEST, SCENE_NAMES_VAL

from training.coarse import eval_epoch as eval_epoch_retrieval
from training.utils import plot_retrievals
from models.superglue_matcher import get_pos_in_cell

import torch_geometric.transforms as T 

'''
RESULTS
- Check top-3 conf -> Not better

TODO:
- Fine: which objects to cut-off? Just pad_size=32 not different, try to select perfectly?
- Fine: select by cluster (2*cell-size) + conf
- Try to add num_matches*10 + sum(match_scores[correctly_matched])
- Care to deactive transform for no-augmentation studies ;)

- How to handle orientation predictions?
'''

@torch.no_grad()
def run_coarse(model, dataloader, args):
    """Run text-to-cell retrieval to obtain the top-cells and coarse pose accuracies.  

    Args:
        model: retrieval model
        dataloader: retrieval dataset
        args: global arguments

    Returns:
        [List]: retrievals as [(cell_indices_i_0, cell_indices_i_1, ...), (cell_indices_i+1, ...), ...] with i ∈ [0, len(poses)-1], j ∈ [0, max(top_k)-1]
        [Dict]: accuracies
    """
    model.eval()

    if args.coarse_oracle: # Build retrievals as [[best_cell, best_cell, ...], [best_cell, best_cell, ...]]
        retrievals = []
        max_k = max(args.top_k)
        for pose in dataloader.dataset.all_poses:
            retrievals.append([pose.cell_id for i in range(max_k)])
    else:
        # Run retrieval model to obtain top-cells
        retrieval_accuracies, retrieval_accuracies_close, retrievals = eval_epoch_retrieval(model, dataloader, args)
        retrievals = [retrievals[idx] for idx in range(len(retrievals))] # Dict -> list
        print('Retrieval Accs:')
        print(retrieval_accuracies)
        print('Retrieval Accs Close:')
        print(retrieval_accuracies_close)    
        assert len(retrievals) == len(dataloader.dataset.all_poses)

    all_cells_dict = {cell.id: cell for cell in dataloader.dataset.all_cells}

    # Gather the accuracies for each sample
    accuracies = {k: {t: [] for t in args.threshs} for k in args.top_k}
    for i_sample in range(len(retrievals)):
        pose = dataloader.dataset.all_poses[i_sample]
        top_cells = [all_cells_dict[cell_id] for cell_id in retrievals[i_sample]]
        pos_in_cells = 0.5 * np.ones((len(top_cells), 2)) # Predict cell-centers
        accs = calc_sample_accuracies(pose, top_cells, pos_in_cells, args.top_k, args.threshs)

        for k in args.top_k:
            for t in args.threshs:
                accuracies[k][t].append(accs[k][t])

    for k in args.top_k:
        for t in args.threshs:
            accuracies[k][t] = np.mean(accuracies[k][t])

    return retrievals, accuracies

def run_fine_oracle(retrievals, dataloader, args):
    assert len(retrievals) == len(dataloader.dataset) == len(dataloader.dataset.all_poses)
    all_cells_dict = {cell.id: cell for cell in dataloader.dataset.all_cells}

    accuracies = {k: {t: [] for t in args.threshs} for k in args.top_k}
    for i_pose, pose in enumerate(dataloader.dataset.all_poses):
        top_cells = [all_cells_dict[cell_id] for cell_id in retrievals[i_pose]]
        
        # Get the ideal in-cell location for each retrieval
        pos_in_cells = []
        for cell in top_cells:
            loc = (pose.pose_w[0:2] - cell.bbox_w[0:2]) / cell.cell_size
            loc = np.clip(loc, 0, 1)
            pos_in_cells.append(loc)
        pos_in_cells = np.array(pos_in_cells)

        accs = calc_sample_accuracies(pose, top_cells, pos_in_cells, args.top_k, args.threshs)

        for k in args.top_k:
            for t in args.threshs:
                accuracies[k][t].append(accs[k][t])
    
    for k in args.top_k:
        for t in args.threshs:
            accuracies[k][t] = np.mean(accuracies[k][t])
    return accuracies

@torch.no_grad()
def run_fine(model, retrievals, dataloader, args):
    # A batch in this dataset contains max(top_k) times the pose vs. each of the max(top_k) top-cells.
    dataset_topk = Kitti360TopKDataset(dataloader.dataset.all_poses, dataloader.dataset.all_cells, retrievals, transform, args)

    num_samples = max(args.top_k)

    t0 = time.time()
    # Obtain the matches, offsets and confidences for each pose vs. its top-cells
    # Using a dataloader does not make it much faster ;)
    matches = []
    offsets = []
    confidences = []
    cell_ids = []
    poses_w = []
    for i_sample, sample in enumerate(dataset_topk):
        output = model(sample['objects'], sample['hint_descriptions'], sample['object_points'])
        matches.append(output.matches0.detach().cpu().numpy())
        offsets.append(output.offsets.detach().cpu().numpy())
        # confs = get_confidences(output.P.detach().cpu().numpy())
        assert len(output.matches0.shape)==2
        out_matches = output.matches0.detach().cpu().numpy()
        # out_match_confs = output.matching_scores0.detach().cpu().numpy()
        confs = np.sum(out_matches >= 0, axis=1)# * 10 + np.sum(out_match_confs[out_ma])
        assert len(confs) == num_samples
        confidences.append(confs)
        
        cell_ids.append([cell.id for cell in sample['cells']])
        poses_w.append(sample['poses'][0].pose_w)

    assert len(matches) == len(offsets) == len(retrievals)
    cell_ids = np.array(cell_ids)

    t1 = time.time()
    print('ela:', t1-t0)

    all_cells_dict = {cell.id: cell for cell in dataloader.dataset.all_cells}

    # Gather the accuracies for each sample
    accuracies_mean = {k: {t: [] for t in args.threshs} for k in args.top_k}
    accuracies_offset = {k: {t: [] for t in args.threshs} for k in args.top_k}
    accuracies_mean_conf = {1: {t: [] for t in args.threshs}}
    for i_sample in range(len(retrievals)):
        pose = dataloader.dataset.all_poses[i_sample]
        top_cells = [all_cells_dict[cell_id] for cell_id in retrievals[i_sample]]
        sample_matches = matches[i_sample]
        sample_offsets = offsets[i_sample]
        sample_confidences = confidences[i_sample]

        if not np.all(np.array([cell.id for cell in top_cells]) == cell_ids[i_sample]):
            print()
            print([cell.id for cell in top_cells])
            print(cell_ids[i_sample])

        assert np.all(np.array([cell.id for cell in top_cells]) == cell_ids[i_sample])
        assert np.allclose(pose.pose_w, poses_w[i_sample])
        
        # Get objects, matches and offsets for each of the top-cells
        pos_in_cells_mean = []
        pos_in_cells_offsets = []
        for i_cell in range(len(top_cells)):
            cell = top_cells[i_cell]
            cell_matches = sample_matches[i_cell]
            cell_offsets = sample_offsets[i_cell]
            pos_in_cells_mean.append(get_pos_in_cell(cell.objects, cell_matches, np.zeros_like(cell_offsets)))
            pos_in_cells_offsets.append(get_pos_in_cell(cell.objects, cell_matches, cell_offsets))
        pos_in_cells_mean = np.array(pos_in_cells_mean)
        pos_in_cells_offsets = np.array(pos_in_cells_offsets)

        accs_mean = calc_sample_accuracies(pose, top_cells, pos_in_cells_mean, args.top_k, args.threshs)
        accs_offsets = calc_sample_accuracies(pose, top_cells, pos_in_cells_offsets, args.top_k, args.threshs)
        
        conf_idx = np.argmax(sample_confidences)
        accs_mean_conf = calc_sample_accuracies(pose, top_cells[conf_idx : conf_idx+1], pos_in_cells_mean[conf_idx : conf_idx+1], top_k=[1,], threshs=args.threshs)

        for k in args.top_k:
            for t in args.threshs:
                accuracies_mean[k][t].append(accs_mean[k][t])        
                accuracies_offset[k][t].append(accs_offsets[k][t])
                accuracies_mean_conf[1][t].append(accs_mean_conf[1][t])

    for k in args.top_k:
        for t in args.threshs:
            accuracies_mean[k][t] = np.mean(accuracies_mean[k][t])
            accuracies_offset[k][t] = np.mean(accuracies_offset[k][t])
            accuracies_mean_conf[1][t] = np.mean(accuracies_mean_conf[1][t])

    return accuracies_mean, accuracies_offset, accuracies_mean_conf


if __name__ == '__main__':
    args = parse_arguments()
    print(str(args).replace(',','\n'), '\n')

    # Load datasets
    transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

    if args.use_validation:
        dataset_retrieval = Kitti360CoarseDatasetMulti(args.base_path, SCENE_NAMES_VAL, transform, shuffle_hints=False, flip_poses=False)    
    else:
        dataset_retrieval = Kitti360CoarseDatasetMulti(args.base_path, SCENE_NAMES_TEST, transform, shuffle_hints=False, flip_poses=False)
        # dataset_retrieval = Kitti360CoarseDatasetMulti(args.base_path, ['2013_05_28_drive_0003_sync', ], transform, shuffle_hints=False, flip_poses=False)
    dataloader_retrieval = DataLoader(dataset_retrieval, batch_size = args.batch_size, collate_fn=Kitti360CoarseDataset.collate_fn)

    # dataset_cell_only = dataset_retrieval.get_cell_dataset()

    # Load models
    model_retrieval = torch.load(args.path_coarse)
    model_matching = torch.load(args.path_fine)

    # eval_conf(model_matching, dataset_retrieval)
    # quit()

    # Run coarse
    retrievals, coarse_accuracies = run_coarse(model_retrieval, dataloader_retrieval, args)
    print_accuracies(coarse_accuracies, "Coarse")

    if args.plot_retrievals:
        plot_retrievals(retrievals, dataset_retrieval)
    if args.plot_retrievals or args.coarse_only:
        quit()

    # Run fine
    if args.fine_oracle:
        accuracies = run_fine_oracle(retrievals, dataloader_retrieval, args)
        print_accuracies(accuracies, "Fine (oracle)")
    else:
        accuracies_mean, accuracies_offsets, accuracies_mean_conf = run_fine(model_matching, retrievals, dataloader_retrieval, args)
        print_accuracies(accuracies_mean, "Fine (mean)")
        print_accuracies(accuracies_offsets, "Fine (offsets)")
        print_accuracies(accuracies_mean_conf, "Fine (mean-conf)")
    
'''
- Re-train cd5 (more memory ;) ) -> Running
- Build cd03, hope that doesn't help anymore -> Prepare running
'''