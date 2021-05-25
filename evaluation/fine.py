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

from training.losses import calc_recall_precision


'''
- Load fine dataset
- Run gather all 5 metrics at once for best cell
'''

@torch.no_grad()
def run_fine(model, dataloader, args):
    stats = EasyDict(
        recall = [],
        precision = [],
        acc_mid = [],
        acc_mean = [],
        acc_offsets = []
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
    assert len(matches) == len(offsets) == len()
            


if __name__ == '__main__':
    print('Evaluating fine model!')
    args = parse_arguments()
    print(str(args).replace(',','\n'), '\n')

    transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])
    dataset_fine = Kitti360FineDatasetMulti(args.base_path, SCENE_NAMES_TEST, transform, args, flip_pose=False)
    dataloader_fine = DataLoader(dataset_fine, batch_size=args.batch_size, collate_fn=Kitti360FineDataset.collate_fn)

    model_matching = torch.load(args.path_fine)

    _ = run_fine()