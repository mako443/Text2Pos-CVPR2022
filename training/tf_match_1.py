import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from models.transformer import TransformerMatch1, get_mlp
from dataloading.semantic3d import Semantic3dObjectReferanceDataset

dataset = Semantic3dObjectReferanceDataset('./data/numpy_merged/', './data/semantic3d')
dataloader = DataLoader(dataset, batch_size=8, collate_fn=Semantic3dObjectReferanceDataset.collate_fn)

# batch = next(iter(dataloader))
# model = TransformerMatch1(dataset.get_known_classes(), dataset.get_known_words(), embedding_dim=32, num_layers=2)
# features, obj_ref_predictions = model(batch['mentioned_objects_classes'], batch['mentioned_objects_positions'], batch['text_descriptions']) #[B, num_obj]
# criterion = nn.BCEWithLogitsLoss() # TODO: weights?

# loss = []
# for i in range(obj_ref_predictions.size(0)):
#     targets = torch.zeros(obj_ref_predictions.size(1))
#     targets[batch['target_idx'][i]] = 1
#     loss.append(criterion(obj_ref_predictions[i], targets))
# loss = torch.mean(torch.stack(loss))

# quit()

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.autograd.set_detect_anomaly(True)

ALPHA_REF, ALPHA_TARGET_CLASS, ALPHA_OBJECT_CLASS = 2.0, 1.0, 1.0

loss_dict = {}
ref_acc_dict = {}
targetclass_acc_dict = {}
objclass_acc_dict = {}
for lr in np.logspace(-2.5,-4,5)[0:3]:
    model = TransformerMatch1(dataset.get_known_classes(), dataset.get_known_words(), embedding_dim=300, num_layers=3)
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_ref = nn.BCEWithLogitsLoss(weight=torch.tensor([5,1,1,1,1,1], dtype=torch.float, device=DEVICE)) # OPTION: weights?
    criterion_target_class = nn.CrossEntropyLoss(weight=torch.tensor([1,1,1,1,1,1], dtype=torch.float, device=DEVICE)) # OPTION: weights?
    criterion_object_class = nn.CrossEntropyLoss()

    loss_key = lr
    loss_dict[loss_key] = []
    ref_acc_dict[loss_key] = []
    targetclass_acc_dict[loss_key] = []
    objclass_acc_dict[loss_key] = []

    for epoch in range(32):
        print(f'\r lr {lr:0.6} epoch {epoch}', end='')
        epoch_losses = []
        epoch_ref_accs = []
        epoch_targetclass_accs = []
        epoch_objclass_accs = []
        for i_batch, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            features, obj_ref_pred, target_class_pred, obj_class_pred = model(batch['mentioned_objects_classes'], batch['mentioned_objects_positions'], batch['text_descriptions']) #[B, num_obj]

            all_preds = []
            known_classes = list(model.known_classes.keys())

            #Obj-ref and obj-class loss
            loss_ref = []
            loss_obj_class = []
            for i in range(obj_ref_pred.size(0)):
                #Obj-ref
                targets = torch.zeros(obj_ref_pred.size(1))
                targets[batch['target_idx'][i]] = 1
                loss_ref.append(criterion_ref(obj_ref_pred[i], targets.to(obj_ref_pred)))
                
                preds = obj_ref_pred[i].cpu().detach().numpy()
                all_preds.append(preds)
                epoch_ref_accs.append(np.argmax(preds) == batch['target_idx'][i])

                #Obj-class
                targets = torch.tensor([known_classes.index(c) for c in batch['mentioned_objects_classes'][i]], dtype=torch.long, device=DEVICE)
                loss_obj_class.append(criterion_object_class(obj_class_pred[i], targets))

                preds = obj_class_pred[i].cpu().detach().numpy()
                epoch_objclass_accs.append(np.mean(np.argmax(preds, axis=1) == targets.cpu().detach().numpy()))

            loss_ref = torch.mean(torch.stack(loss_ref))
            loss_obj_class = torch.mean(torch.stack(loss_obj_class))
            
            #Target class loss
            target_class_indices = torch.tensor([list(model.known_classes.keys()).index(batch['target_classes'][i]) for i in range(obj_ref_pred.size(0))], dtype=torch.long, device=DEVICE)
            loss_target_class = criterion_target_class(target_class_pred, target_class_indices)
            preds = target_class_pred.cpu().detach().numpy()
            epoch_targetclass_accs.append(np.mean(np.argmax(preds, axis=1) == target_class_indices.cpu().detach().numpy()))

            loss = ALPHA_REF * loss_ref + ALPHA_TARGET_CLASS * loss_target_class + ALPHA_OBJECT_CLASS * loss_obj_class

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            
            # if i_batch==3:
            #     break

        loss_dict[loss_key].append(np.mean(epoch_losses))
        ref_acc_dict[loss_key].append(np.mean(epoch_ref_accs))
        targetclass_acc_dict[loss_key].append(np.mean(epoch_targetclass_accs))
        objclass_acc_dict[loss_key].append(np.mean(epoch_objclass_accs))

    print()
    print(preds)
    print(target_class_indices)
    print()

plt.subplot(2,2,1)
for k in loss_dict.keys():
    l=loss_dict[k]
    line, = plt.plot(l)
    line.set_label(f'{k:0.6}')
plt.title('Losses')
plt.gca().set_ylim(bottom=0.0) #Set the bottom to 0.0
plt.legend()

plt.subplot(2,2,2)
for k in ref_acc_dict.keys():
    l=ref_acc_dict[k]
    line, = plt.plot(l)
    line.set_label(f'{k:0.6}')
plt.title('Ref-Accuracy')
plt.gca().set_ylim(bottom=0.0) #Set the bottom to 0.0
plt.legend()

plt.subplot(2,2,3)
for k in targetclass_acc_dict.keys():
    l=targetclass_acc_dict[k]
    line, = plt.plot(l)
    line.set_label(f'{k:0.6}')
plt.title('Targetclass-Accuracy')
plt.gca().set_ylim(bottom=0.0) #Set the bottom to 0.0
plt.legend()

plt.subplot(2,2,4)
for k in objclass_acc_dict.keys():
    l=objclass_acc_dict[k]
    line, = plt.plot(l)
    line.set_label(f'{k:0.6}')
plt.title('Objectclass-Accuracy')
plt.gca().set_ylim(bottom=0.0) #Set the bottom to 0.0
plt.legend()

plt.show()
