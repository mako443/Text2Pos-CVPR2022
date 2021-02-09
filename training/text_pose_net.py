import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import string
import random
import os
import os.path as osp
import sys
import numpy as np
import matplotlib.pyplot as plt

from dataloading.textdescriptions import TextDescriptionData
from models.text_pose_net import TextPoseNet
from evaluation.eval import eval_pose_predictions

'''
TODO:
- shuffle sentences in description => Actually needs other descriptions structure to also shuffle the order of objects etc. ✖
'''

BATCH_SIZE = 16
EMBED_DIM = 512
BETA = 500

print(f'TextPoseNet train: embed-dim {EMBED_DIM}, batch-size {BATCH_SIZE}, beta {BETA}')

train_mask = np.random.randint(6, size=160)>0
val_mask   = np.invert(train_mask)

data_set = TextDescriptionData('data/semantic3d', 'train', 'sg27_station5_intensity_rgb', split_mask=train_mask)
data_set_val = TextDescriptionData('data/semantic3d', 'train', 'sg27_station5_intensity_rgb', split_mask=val_mask)
data_loader=DataLoader(data_set, batch_size=BATCH_SIZE, num_workers=2, pin_memory=False, shuffle=True) 

loss_dict={}
best_loss=np.inf
best_model=None

for lr in (5e-3, 1e-3, 5e-4):
    print('\n\nlr: ',lr)

    model = TextPoseNet(data_set.get_known_words(), EMBED_DIM)
    model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr= lr) #Adam is ok for PyG | Apparently also for packed_sequence!

    loss_dict[lr] = []
    for epoch in range(32):
        epoch_loss_sum=0.0
        for i_batch, batch in enumerate(data_loader):
            optimizer.zero_grad()            
            
            out = model(batch['descriptions'])
            target = batch['poses'].cuda()

            loss = criterion(out[:, 0:2], target[:, 0:2]) + BETA*criterion(out[:, 2:6], target[:, 2:6])

            loss.backward()
            optimizer.step()

            l=loss.cpu().detach().numpy()
            epoch_loss_sum+=l
            #print(f'\r epoch {epoch} loss {l}',end='')

        epoch_avg_loss = epoch_loss_sum/(i_batch+1)
        print(f'epoch {epoch} final avg-loss {epoch_avg_loss}')
        loss_dict[lr].append(epoch_avg_loss)

    #Now using loss-avg of last epoch!
    if epoch_avg_loss<best_loss:
        best_loss=epoch_avg_loss
        best_model=model

    accuracy = eval_pose_predictions(data_set, model)
    print(accuracy)
    accuracy = eval_pose_predictions(data_set_val, model)
    print(accuracy)    

print('\n----')           
model_name = f'model_TextPoseNet_b{BATCH_SIZE}_e{EMBED_DIM}_b{BETA}'
print('Saving best model',model_name)
torch.save(best_model.state_dict(),f'model_{model_name}.pth')

for k in loss_dict.keys():
    l=loss_dict[k]
    line, = plt.plot(l)
    line.set_label(k)
plt.gca().set_ylim(bottom=0.0) #Set the bottom to 0.0
plt.legend()
#plt.show()
plt.savefig(f'loss_{model_name}.png')    
