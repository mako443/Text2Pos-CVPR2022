from typing import List

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict

from datapreparation.kitti360.imports import Object3d, Pose, Cell

from models.superglue_matcher import get_pos_in_cell, get_pos_in_cell_intersect

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

def calc_recall_precision(batch_gt_matches, batch_matches0, batch_matches1):
    assert len(batch_gt_matches) == len(batch_matches0) == len(batch_matches1)
    all_recalls = []
    all_precisions = []

    for idx in range(len(batch_gt_matches)):
        gt_matches, matches0, matches1 = batch_gt_matches[idx], batch_matches0[idx], batch_matches1[idx]
        gt_matches = gt_matches.tolist()

        recall = []
        for i,j in gt_matches:
            recall.append(matches0[i] == j or matches1[j] == i)

        precision = []
        for i,j in enumerate(matches0):
            if j>= 0:
                precision.append([i, j] in gt_matches) #CARE: this only works as expected after tolist()

        recall = np.mean(recall) if len(recall)>0 else 0.0
        precision = np.mean(precision) if len(precision)>0 else 0.0
        all_recalls.append(recall)
        all_precisions.append(precision)

    return np.mean(all_recalls), np.mean(all_precisions)

def calc_pose_error_intersect(objects, matches0, poses: List[Pose], directions):
    assert len(objects) == len(matches0) == len(poses)
    assert isinstance(poses[0], Pose)

    batch_size, pad_size = matches0.shape
    poses = np.array([pose.pose for pose in poses])[:, 0:2] # Assuming this is the best cell!

    errors = []
    for i_sample in range(batch_size):
        pose_prediction = get_pos_in_cell_intersect(objects[i_sample], matches0[i_sample], directions[i_sample])
        errors.append(np.linalg.norm(poses[i_sample] - pose_prediction))
    return np.mean(errors)

# Verified against old function ✓
def calc_pose_error(objects, matches0, poses: List[Pose], offsets=None, use_mid_pred=False):
    """Calculates the mean error of a batch by averaging the positions of all matches objects plus corresp. offsets.
    All calculations are in x-y-plane.

    Args:
        objects (List[List[Object3D]]): Objects-list for each sample in the batch.
        matches0 (np.ndarray): SuperGlue matching output of the batch.
        poses (np.ndarray): Ground-truth poses [batch_size, 3]
        offsets (List[np.ndarray], optional): List of offset vectors for all hints. Zero offsets are used if not given.
        use_mid_pred (bool, optional): If set, predicts the center of the cell regardless of matches and offsets. Defaults to False.

    Returns:
        [float]: Mean error.
    """
    assert len(objects) == len(matches0) == len(poses)
    assert isinstance(poses[0], Pose)

    batch_size, pad_size = matches0.shape
    poses = np.array([pose.pose for pose in poses])[:, 0:2] # Assuming this is the best cell!

    if offsets is not None:
        assert len(objects) == len(offsets)     
    else:
        offsets = np.zeros((batch_size, pad_size, 2)) # Set zero offsets to just predict the mean of matched-objects' centers

    errors = []
    for i_sample in range(batch_size):
        if use_mid_pred:
            pose_prediction = np.array((0.5, 0.5))
        else:
            pose_prediction = get_pos_in_cell(objects[i_sample], matches0[i_sample], offsets[i_sample])
        errors.append(np.linalg.norm(poses[i_sample] - pose_prediction))
    return np.mean(errors)
        

def deprecated_calc_pose_error(objects, matches0, poses, args, offsets=None, use_mid_pred=False):
    """Calculates the mean error by adding offset ("obj-to-pose") to every corresponding, matched object
    Uses simple matched-objects-average if offsets not given
    CARE: error only in x-y-plane

    Args:
        objects
        matches0: matches0[i] = [j] <-> object[i] matches hint/offset[j]]
        poses
        offsets (optional): Predicted offsets. Defaults to None.
        use_mid_pred (optional): Use cell-mid (0.5,0.5) as prediction, discarding objects and matches (for debugging).

    Returns:
        [float]: mean error
    """
    assert len(objects) == len(matches0) == len(poses)
    batch_size, pad_size = matches0.shape
    poses = np.array(poses)[:, 0:2]

    if offsets is not None:
        assert len(objects) == len(offsets)     
    else:
        offsets = np.zeros((batch_size, pad_size, 2)) # Set zero offsets to just predict the mean of matched-objects' centers

    errors = []
    for i_sample in range(batch_size):
        preds = []
        for obj_idx, hint_idx in enumerate(matches0[i_sample]):
            if hint_idx == -1:
                continue
            if args.dataset == 'S3D':
                preds.append(objects[i_sample][obj_idx].center[0:2] + offsets[i_sample][hint_idx])
            else:
                preds.append(objects[i_sample][obj_idx].closest_point[0:2] + offsets[i_sample][hint_idx])
        if use_mid_pred:
            pose_prediction = np.array((0.5,0.5))
        else:
            pose_prediction = np.mean(preds, axis=0) if len(preds)>0 else np.array((0.5,0.5)) # Guess the middle if no matches
        errors.append(np.linalg.norm(poses[i_sample] - pose_prediction))
    return np.mean(errors)

class PairwiseRankingLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, im, s): #Norming the input (as in paper) is actually not helpful
        im=im/torch.norm(im,dim=1,keepdim=True)
        s=s/torch.norm(s,dim=1,keepdim=True)

        margin = self.margin
        # compute image-sentence score matrix
        scores = torch.mm(im, s.transpose(1, 0))
        #print(scores)
        diagonal = scores.diag()

        # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
        cost_s = torch.max(torch.autograd.Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()), (margin-diagonal).expand_as(scores)+scores)
        # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
        cost_im = torch.max(torch.autograd.Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()), (margin-diagonal).expand_as(scores).transpose(1, 0)+scores)

        for i in range(scores.size()[0]):
            cost_s[i, i] = 0
            cost_im[i, i] = 0

        return (cost_s.sum() + cost_im.sum()) / len(im) #Take mean for batch-size stability     

#My implementation, sanity check done ✓
class HardestRankingLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(HardestRankingLoss, self).__init__()
        self.margin=margin
        self.relu=nn.ReLU()

    def forward(self, images, captions):
        assert images.shape==captions.shape and len(images.shape)==2
        images=images/torch.norm(images,dim=1,keepdim=True)
        captions=captions/torch.norm(captions,dim=1,keepdim=True)        
        num_samples=len(images)

        similarity_scores = torch.mm( images, captions.transpose(1,0) ) # [I x C]

        cost_images= self.margin + similarity_scores - similarity_scores.diag().view((num_samples,1))
        cost_images.fill_diagonal_(0)
        cost_images=self.relu(cost_images)
        cost_images,_=torch.max(cost_images, dim=1)
        cost_images=torch.mean(cost_images)

        cost_captions= self.margin + similarity_scores.transpose(1,0) - similarity_scores.diag().view((num_samples,1))
        cost_captions.fill_diagonal_(0)
        cost_captions=self.relu(cost_captions)
        cost_captions,_=torch.max(cost_captions, dim=1)
        cost_captions=torch.mean(cost_captions)        

        cost= cost_images+cost_captions      
        return cost     

if __name__ == '__main__':
    objects = [
        [EasyDict(center=np.array([0,0])), EasyDict(center=np.array([10,10])), EasyDict(center=np.array([99,99]))],
    ]
    matches0 = np.array((0, 1, -1)).reshape((1,3))
    poses = np.array((0,10)).reshape((1,2))
    offsets = np.array([(2,10), (-10,0)]).reshape((1,2,2))

    err = calc_pose_error(objects, matches0, poses, offsets=None)
    print(err)
    err = calc_pose_error(objects, matches0, poses, offsets=offsets)
    print(err)    
