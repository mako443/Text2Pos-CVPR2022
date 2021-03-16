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