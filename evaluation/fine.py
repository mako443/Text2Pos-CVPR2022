from datapreparation.kitti360.imports import Object3d
from models.superglue_matcher import get_pos_in_cell
import numpy as np
from easydict import EasyDict
import time

import torch
from torch.utils.data import DataLoader
import torch_geometric.transforms as T 

from evaluation.args import parse_arguments
from evaluation.utils import calc_sample_accuracies, print_accuracies

from dataloading.kitti360.poses import Kitti360FineDataset, Kitti360FineDatasetMulti
from datapreparation.kitti360.utils import SCENE_NAMES_TEST
# from training.fine import eval_epoch as eval_epoch_fine
from training.losses import calc_pose_error

from training.losses import calc_recall_precision


'''
- Load fine dataset
- Run gather all 5 metrics at once for best cell
'''

@torch.no_grad()
def run_fine(model, dataloader):
    stats = EasyDict(
        recall = [],
        precision = [],
        mid = [],
        mean = [],
        offsets = [],
        matching_oracle = [],
        offset_oracle = [],
        both_oracle = []
    )

    for i_batch, batch in enumerate(dataloader):
        output = model(batch['objects'], batch['hint_descriptions'], batch['object_points'])
        
        for key in output:
            output[key] = output[key].cpu().detach().numpy()

        recall, precision = calc_recall_precision(batch['matches'], output.matches0, output.matches1)
        stats.recall.append(recall)
        stats.precision.append(precision)

        stats.mid.append(calc_pose_error(batch['objects'], output.matches0, batch['poses'], output.offsets, use_mid_pred=True))
        stats.mean.append(calc_pose_error(batch['objects'], output.matches0, batch['poses'], np.zeros_like(output.offsets)))
        stats.offsets.append(calc_pose_error(batch['objects'], output.matches0, batch['poses'], output.offsets))

        # Build gt-matches
        gt_matches = np.ones_like(output.matches0) * -1 # -1 to put all as negative
        for i_sample, sample_matches in enumerate(batch['matches']):
            for (obj_idx, hint_idx) in sample_matches:
                gt_matches[i_sample][obj_idx] = hint_idx
        stats.matching_oracle.append(calc_pose_error(batch['objects'], gt_matches, batch['poses'], output.offsets))

        # Use gt-offsets
        stats.offset_oracle.append(calc_pose_error(batch['objects'], output.matches0, batch['poses'], batch['offsets']))

        stats.both_oracle.append(calc_pose_error(batch['objects'], gt_matches, batch['poses'], batch['offsets']))
    
    for key in stats:
        stats[key] = np.mean(stats[key])
    return stats



@torch.no_grad()
def depr_run_fine(model, dataloader, args):
    stats = EasyDict(
        recall = [],
        precision = [],
        acc_mid = [],
        acc_mean = [],
        acc_offsets = [],
        acc_matching_oracle = [],
        acc_offsets_oracle = [],
        acc_all_oracle = [],
    )

    # Gather matches, offsets and recall/precision for all samples
    matches = []
    offsets = []
    for i_batch, batch in enumerate(dataloader):
        output = model(batch['objects'], batch['hint_descriptions'], batch['object_points'])
        
        matches.append(output.matches0.cpu().detach().numpy())
        offsets.append(output.offsets.detach().cpu().numpy())
        recall, precision = calc_recall_precision(batch['matches'], output.matches0.cpu().detach().numpy(), output.matches1.cpu().detach().numpy())
        stats.recall.append(recall)
        stats.precision.append(precision)

    matches = np.vstack(matches)
    offsets = np.vstack(offsets)
    assert len(matches) == len(offsets) == len(dataloader.dataset)

    # For each sample, gather the 5 accuracies
    for i_sample in range(len(matches)):
        data = dataloader.dataset[i_sample]
        pose = data['poses']
        cell = data['cells']
        assert cell.id == pose.cell_id

        # Pad the objects: The matching model might have matched a padding object
        cell_objects = cell.objects
        while len(cell_objects) < args.pad_size:
            cell_objects.append(Object3d.create_padding())

        # sample_matches = matches[i_sample][0 : args.pad_size] # Cut-off the matches as well. NOTE: This makes the matching-oracle slightly imperfect

        pos_mid = np.array((0.5, 0.5))
        pos_mean = get_pos_in_cell(cell_objects, matches[i_sample], np.zeros_like(offsets[i_sample]))

        pos_offsets = get_pos_in_cell(cell_objects, matches[i_sample], offsets[i_sample])

        # Build matching oracle
        gt_matches = -1 * np.ones_like(matches[i_sample]) # Set to -1 for unmatched
        for (obj_idx, hint_idx) in data['matches']:
            if obj_idx < len(matches[i_sample]):
                gt_matches[obj_idx] = hint_idx
        pos_matching_oracle = get_pos_in_cell(cell_objects, gt_matches, offsets[i_sample])
    
        # Build offset oracle
        pos_offsets_oracle = get_pos_in_cell(cell_objects, matches[i_sample], data['offsets'])

        pos_all_oracle = get_pos_in_cell(cell_objects, gt_matches, data['offsets'])

        # Get the target
        target = (pose.pose_w[0:2] - cell.bbox_w[0:2]) / cell.cell_size
        assert np.all(target <= 1.0) and np.all(target >= 0.0)

        # Gather the accuracies
        stats.acc_mid.append(np.linalg.norm(target - pos_mid))
        stats.acc_mean.append(np.linalg.norm(target - pos_mean))
        stats.acc_offsets.append(np.linalg.norm(target - pos_offsets))
        stats.acc_matching_oracle.append(np.linalg.norm(target - pos_matching_oracle))
        stats.acc_offsets_oracle.append(np.linalg.norm(target - pos_offsets_oracle))
        stats.acc_all_oracle.append(np.linalg.norm(target - pos_all_oracle))

    for key in stats.keys():
        stats[key] = np.mean(stats[key])
    return stats


if __name__ == '__main__':
    args = parse_arguments()
    print(str(args).replace(',','\n'), '\n')

    if args.no_pc_augment:
        transform = T.FixedPoints(args.pointnet_numpoints)
    else:
        transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])  

    dataset_fine = Kitti360FineDatasetMulti(args.base_path, SCENE_NAMES_TEST, transform, args, flip_pose=False)
    # dataset_fine = Kitti360FineDatasetMulti(args.base_path, ['2013_05_28_drive_0003_sync', ], transform, args, flip_pose=False)
    dataloader_fine = DataLoader(dataset_fine, batch_size=args.batch_size, collate_fn=Kitti360FineDataset.collate_fn)

    model_matching = torch.load(args.path_fine)

    stats = run_fine(model_matching, dataloader_fine)
    for key in stats:
        print(f'{key}: {stats[key]:0.3}')