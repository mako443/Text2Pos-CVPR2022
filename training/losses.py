import numpy as np
import torch
import torch.nn as nn

class MatchingLoss(nn.Module):
    def __init__(self):
        super(MatchingLoss, self).__init__()
        self.eps = 1e-3

    #Matches as list[tensor ∈ [M1, 2], tensor ∈ [M2, 2], ...] for Mi: i ∈ [1, batch_size]
    def forward(self, P, all_matches):
        assert len(P.shape)==3
        assert len(all_matches[0].shape)==2 and all_matches[0].shape[-1]==2
        assert len(P) == len(all_matches)
        batch_losses = []
        for i in range(len(all_matches)):
            matches = all_matches[i]
            cell_losses = -torch.log(P[i, matches[:,0], matches[:, 1]])
            batch_losses.append(torch.mean(cell_losses))

        return torch.mean(torch.stack(batch_losses))

def calc_recall_precision(gt_matches, matches0, matches1):
    assert gt_matches.shape[1]==2 and len(gt_matches.shape)==2
    gt_matches = gt_matches.tolist()

    recall = []
    for i,j in gt_matches:
        recall.append(matches0[i] == j or matches1[j] == i)

    precision = []
    for i,j in enumerate(matches0):
        if j>= 0:
            precision.append([i, j] in gt_matches) #CARE: this only works as expected after tolist()

    return np.mean(recall), np.mean(precision)