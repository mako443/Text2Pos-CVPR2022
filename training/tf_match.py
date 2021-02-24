import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from models.transformer import TransformerMatch1
from dataloading.semantic3d import Semantic3dObjectReferanceDataset

from training.args import parse_arguments
from training.plots import plot_metrics

def train_epoch(model, dataloader, args):
    known_classes = list(model.known_classes.keys())

    epoch_losses = []
    epoch_acc_ref = []
    epoch_acc_tgtclass = []
    epoch_acc_objclass = []
    epoch_acc_offset = []

    for i_batch, batch in enumerate(dataloader):
        if args.max_batches is not None and i_batch >= args.max_batches:
            break

        optimizer.zero_grad()
        model_output = model(batch['objects_classes'], batch['objects_positions'], batch['text_descriptions']) #[B, num_obj]

        #Obj-ref and obj-class loss, TODO: vectorize
        loss_ref = []
        loss_objclass = []
        loss_offset = []
        for i in range(model_output.obj_ref_pred.size(0)):
            #Obj-ref
            targets = torch.zeros(model_output.obj_ref_pred.size(1))
            targets[batch['target_idx'][i]] = 1
            assert batch['target_idx'][i] == 0
            loss_ref.append(criterion_ref(model_output.obj_ref_pred[i], targets.to(model_output.obj_ref_pred)))
            
            preds = model_output.obj_ref_pred[i].cpu().detach().numpy()
            epoch_acc_ref.append(np.argmax(preds) == batch['target_idx'][i])

            #Obj-class
            targets = torch.tensor([known_classes.index(c) for c in batch['objects_classes'][i]], dtype=torch.long, device=DEVICE)
            loss_objclass.append(criterion_object_class(model_output.obj_class_pred[i], targets))

            preds = model_output.obj_class_pred[i].cpu().detach().numpy()
            epoch_acc_objclass.append(np.mean(np.argmax(preds, axis=1) == targets.cpu().detach().numpy()))

            #Offset
            indices = torch.arange(1, batch['description_lengths'][i]) # Select the mentioned, non-target objects
            inputs = model_output.obj_offset_pred[i][indices]
            targets = torch.tensor(batch['offset_vectors'][i][indices], dtype=torch.float, device=DEVICE)
            loss_offset.append(criterion_offset(inputs, targets))

            indices = indices.cpu().detach().numpy()
            preds = model_output.obj_offset_pred[i].cpu().detach().numpy()
            diffs = preds[indices] - batch['offset_vectors'][i][indices]
            diffs = np.linalg.norm(diffs, axis=1)
            epoch_acc_offset.append(np.mean(diffs))

        loss_ref = torch.mean(torch.stack(loss_ref))
        loss_objclass = torch.mean(torch.stack(loss_objclass))
        loss_offset = torch.mean(torch.stack(loss_offset))

        #Target class loss
        target_class_indices = torch.tensor([list(model.known_classes.keys()).index(batch['target_classes'][i]) for i in range(model_output.obj_ref_pred.size(0))], dtype=torch.long, device=DEVICE)
        loss_target_class = criterion_target_class(model_output.target_class_pred, target_class_indices)
        preds = model_output.target_class_pred.cpu().detach().numpy()
        epoch_acc_tgtclass.append(np.mean(np.argmax(preds, axis=1) == target_class_indices.cpu().detach().numpy()))

        loss = ALPHA_REF * loss_ref + ALPHA_TARGET_CLASS * loss_target_class + ALPHA_OBJECT_CLASS * loss_objclass #+ ALPHA_OFFSET * loss_offset
        
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    return np.mean(epoch_losses), np.mean(epoch_acc_ref), np.mean(epoch_acc_tgtclass), np.mean(epoch_acc_objclass), np.mean(epoch_acc_offset)                    

@torch.no_grad()
def val_epoch(model, dataloader, args):
    known_classes = list(model.known_classes.keys())

    epoch_acc_ref = []
    epoch_acc_tgtclass = []
    epoch_acc_objclass = []
    epoch_acc_offset = []

    for i_batch, batch in enumerate(dataloader):
        model_output = model(batch['objects_classes'], batch['objects_positions'], batch['text_descriptions']) #[B, num_obj]

        #Obj-ref and obj-class loss, TODO: vectorize
        loss_objclass = []
        for i in range(model_output.obj_ref_pred.size(0)):
            #Obj-ref            
            preds = model_output.obj_ref_pred[i].cpu().detach().numpy()
            epoch_acc_ref.append(np.argmax(preds) == batch['target_idx'][i])

            #Obj-class
            targets = torch.tensor([known_classes.index(c) for c in batch['objects_classes'][i]], dtype=torch.long, device=DEVICE)
            preds = model_output.obj_class_pred[i].cpu().detach().numpy()
            epoch_acc_objclass.append(np.mean(np.argmax(preds, axis=1) == targets.cpu().detach().numpy()))

            #Offset
            indices = torch.arange(1, batch['description_lengths'][i]) # Select the mentioned, non-target objects

            indices = indices.cpu().detach().numpy()
            preds = model_output.obj_offset_pred[i].cpu().detach().numpy()
            diffs = preds[indices] - batch['offset_vectors'][i][indices]
            diffs = np.linalg.norm(diffs, axis=1)
            epoch_acc_offset.append(np.mean(diffs))            

        #Target class
        target_class_indices = torch.tensor([list(model.known_classes.keys()).index(batch['target_classes'][i]) for i in range(model_output.obj_ref_pred.size(0))], dtype=torch.long, device=DEVICE)
        preds = model_output.target_class_pred.cpu().detach().numpy()
        epoch_acc_tgtclass.append(np.mean(np.argmax(preds, axis=1) == target_class_indices.cpu().detach().numpy()))

    return np.mean(epoch_acc_ref), np.mean(epoch_acc_tgtclass), np.mean(epoch_acc_objclass), np.mean(epoch_acc_offset)

if __name__ == "__main__":
    args = parse_arguments()
    print(args, '\n')

    '''
    Create data loaders
    '''
    dataset_train = Semantic3dObjectReferanceDataset('./data/numpy_merged/', './data/semantic3d', num_distractors=args.num_distractors, split='train')
    dataloader_train = DataLoader(dataset_train, batch_size=8, collate_fn=Semantic3dObjectReferanceDataset.collate_fn)
    data0 = dataset_train[0]

    dataset_val = Semantic3dObjectReferanceDataset('./data/numpy_merged/', './data/semantic3d', num_distractors=args.num_distractors, split='test')
    dataloader_val = DataLoader(dataset_val, batch_size=8, collate_fn=Semantic3dObjectReferanceDataset.collate_fn)

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.autograd.set_detect_anomaly(True)

    ALPHA_REF = args.alpha_obj_ref
    ALPHA_TARGET_CLASS = args.alpha_target_class
    ALPHA_OBJECT_CLASS = args.alpha_obj_class
    # ALPHA_OFFSET = 0.01    

    '''
    Start training
    '''
    learning_rates = np.logspace(-2.5,-4,5)[1:]

    dict_loss = {lr: [] for lr in learning_rates}
    dict_acc_ref = {lr: [] for lr in learning_rates}
    dict_acc_tgtclass = {lr: [] for lr in learning_rates}
    dict_acc_objclass = {lr: [] for lr in learning_rates}
    dict_acc_offset = {lr: [] for lr in learning_rates}

    dict_acc_ref_val = {lr: [] for lr in learning_rates}
    dict_acc_tgtclass_val = {lr: [] for lr in learning_rates}
    dict_acc_objclass_val = {lr: [] for lr in learning_rates}
    dict_acc_offset_val = {lr: [] for lr in learning_rates}

    for lr in learning_rates:
        model = TransformerMatch1(dataset_train.get_known_classes(), dataset_train.get_known_words(), embedding_dim=args.embed_dim, num_layers=args.num_layers)
        model = model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,args.lr_gamma)

        criterion_ref_weights = torch.ones(len(data0['objects_classes']), dtype=torch.float, device=DEVICE) #Balance the weight of one ref-target vs. #mentioned + #distractor -1 other objects
        criterion_ref_weights[0] = len(data0['objects_classes'])-1
        criterion_ref = nn.BCEWithLogitsLoss(weight=criterion_ref_weights) # OPTION: weights?
        criterion_target_class = nn.CrossEntropyLoss(weight=torch.tensor([1,1,1,1,1,1], dtype=torch.float, device=DEVICE)) # OPTION: weights?
        criterion_object_class = nn.CrossEntropyLoss()
        criterion_offset = nn.MSELoss()

        loss_key = lr   

        for epoch in range(32):
            print(f'\r lr {lr:0.6} epoch {epoch}', end='')

            loss, acc_ref, acc_tgtclass, acc_objclass, acc_offset = train_epoch(model, dataloader_train, args)
            scheduler.step()
            dict_loss[loss_key].append(loss)
            dict_acc_ref[loss_key].append(acc_ref)
            dict_acc_tgtclass[loss_key].append(acc_tgtclass)
            dict_acc_objclass[loss_key].append(acc_objclass)     
            dict_acc_offset[loss_key].append(acc_offset)

            acc_ref, acc_tgtclass, acc_objclass, acc_offset = val_epoch(model, dataloader_val, args)
            dict_acc_ref_val[loss_key].append(acc_ref)
            dict_acc_tgtclass_val[loss_key].append(acc_tgtclass)
            dict_acc_objclass_val[loss_key].append(acc_objclass)      
            dict_acc_offset_val[loss_key].append(acc_offset)  

        print()

    '''
    Save plots
    '''
    plot_name = f'e{args.embed_dim}_l{args.num_layers}_dist{args.num_distractors}_batches{args.max_batches}_gamma{args.lr_gamma}.png'
    metrics = {
        'train-loss': dict_loss,
        'train-acc-ref': dict_acc_ref,
        'train-acc-targetclass': dict_acc_tgtclass,
        'train-acc-objclass': dict_acc_objclass,
        'val-acc-ref': dict_acc_ref_val,
        'val-acc-targetclass': dict_acc_tgtclass_val,
        'val-acc-objclass': dict_acc_objclass_val,
    }
    plot_metrics(metrics, './plots/'+plot_name)