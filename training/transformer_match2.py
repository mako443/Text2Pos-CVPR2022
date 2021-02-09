import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataloading.listdescriptions import MockData
from dataloading.semantic3d import Semantic3dObjectData

'''
TODO:
- Currently no padding / assumes equal lengths
- Loss to all/hardest, not (random) negative
- If L2-distances not ok: regress TxS softmax matching matrix using additional shared MLP?
'''

EMBED_DIM = 128
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

### Mock
CLASSES = ('red', 'green', 'blue', 'yellow', '<pad>', '<tok>')
### S3d
# CLASSES = ('high vegetation', 'low vegetation', 'buildings', 'hard scape', 'cars', '<pad>', '<tok>')

DIRECTIONS = ('left','right','ahead','behind', '<pad>', '<tok>')


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=4, dim_feedforward=2*EMBED_DIM)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 4) #TODO: Add norm?

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=EMBED_DIM, nhead=4, dim_feedforward=2*EMBED_DIM)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, 4)

        self.classes = CLASSES
        self.class_embedding = nn.Embedding(len(self.classes), EMBED_DIM)
        self.directions = DIRECTIONS
        self.direction_embedding = nn.Embedding(len(self.directions), EMBED_DIM)

        self.position_embedding = nn.Linear(2, EMBED_DIM)

        self.mlp0 = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.mlp1 = nn.Linear(EMBED_DIM, 1)

    def forward(self, batch_object_classes, batch_object_positions, batch_description_classes, batch_description_directions):
        #Append tokens
        for i_batch in range(len(batch_object_classes)):
            batch_description_classes[i_batch].append('<tok>')
            batch_description_directions[i_batch].append('<tok>')

        object_classes = torch.stack([ self.class_embedding(torch.tensor([self.classes.index(c) for c in object_classes], device=DEVICE)) for object_classes in batch_object_classes ])
        object_positions = self.position_embedding( torch.stack([torch.tensor(object_positions, dtype=torch.float, device=DEVICE) for object_positions in batch_object_positions]) )

        description_classes = torch.stack([ self.class_embedding(torch.tensor([self.classes.index(c) for c in description_classes], device=DEVICE)) for description_classes in batch_description_classes ])
        description_directions = torch.stack([ self.class_embedding(torch.tensor([self.directions.index(c) for c in description_directions], device=DEVICE)) for description_directions in batch_description_directions ])

        object_classes = torch.transpose(object_classes, 0, 1)
        object_positions = torch.transpose(object_positions, 0, 1)
        description_classes = torch.transpose(description_classes, 0, 1)
        description_directions = torch.transpose(description_directions, 0, 1)

        encoded = self.encoder(object_classes + object_positions) # [S,N,E]
        decoded = self.decoder(description_classes + description_directions, encoded) # [T,N,E]

        #Regress angle based on last <tok> index of decoded
        angles = self.mlp1(F.relu(self.mlp0(decoded[-1, :, :])))

        #Do not return the <tok> indices in decoded
        return encoded, decoded[0:-1], angles

'''
TODO: attend to padding and token
Written explicitly for better readability
Assumes encoded and decoded will be transposed and reshaped to [N*S, EMBED] and [N*T, EMBED]
Anchor-Decoded, Positive-Encoded, Negative-Encoded
'''
def build_triplet_indices(match_indices, N, S, T):
    anchor_indices = []
    positive_indices = []
    negative_indices = []
    match_indices = np.array(match_indices)
    # assert match_indices.shape==(N,T), (match_indices.shape, N, T)

    for i in range(N):
        for j in range(T):
            #Exclude padding
            if j>=len(match_indices[i]):
                break

            offset_obj = i*S
            offset_des = i*T
            anchor_indices.append(offset_des + j)
            positive_indices.append(offset_obj + match_indices[i,j])
            negative_indices.append(offset_obj + np.random.choice([idx for idx in range(S) if idx!=match_indices[i,j]]))

    return anchor_indices, positive_indices, negative_indices

def train_epoch():
    model.train()

    epoch_losses = []
    for data in dataloader:
        optimizer.zero_grad()

        encoded, decoded, angles = model(data['object_classes'], data['object_positions'], data['description_classes'], data['description_directions']) #Encoded: [S,N,E], Decoded: [T,N,E]
        encoded = torch.transpose(encoded, 0, 1) # [N,S,E]
        decoded = torch.transpose(decoded, 0, 1) # [N,T,E]
        
        #Norm seems more stable...?
        F.normalize(encoded, dim=-1)
        F.normalize(decoded, dim=-1)

        N, S, E = encoded.shape
        N, T, E = decoded.shape
        a, p, n = build_triplet_indices(data['match_indices'], N, S, T)

        anchor, positive, negative = decoded.reshape((-1, EMBED_DIM))[a], encoded.reshape((-1, EMBED_DIM))[p], encoded.reshape((-1, EMBED_DIM))[n]
        
        loss_corresp = criterion(anchor, positive, negative)
        loss_angle = criterion_angle(angles.flatten(), torch.tensor(data['angles'], dtype=torch.float32, device=DEVICE))
        loss = loss_corresp #+ 1/10.0*loss_angle
        loss.backward()
        optimizer.step()    

        epoch_losses.append(loss.cpu().detach().numpy())
    return np.mean(epoch_losses)

@torch.no_grad()
def test():
    model.eval()

    epoch_accuracies = []
    epoch_angles = []
    for i_data in range(len(dataset)):
        data = dataset[i_data]
        encoded, decoded, angles = model([data['object_classes'], ], [data['object_positions'], ], [data['description_classes'], ], [data['description_directions'], ])
        encoded = torch.transpose(encoded, 0, 1) # [1,S,E]
        decoded = torch.transpose(decoded, 0, 1) # [1,T,E]
        encoded = encoded[0].detach().cpu().numpy()
        decoded = decoded[0].detach().cpu().numpy()
        angles = angles.flatten()[0].detach().cpu().numpy()
        #Norm like in training
        encoded /= np.linalg.norm(encoded)
        decoded /= np.linalg.norm(decoded)
        
        accuracies = []
        acc_angles = []
        for i in range(len(decoded)):
            #Exclude padding
            if i>=len(data['match_indices']):
                break
            diffs = np.linalg.norm(decoded[i] - encoded, axis=1)
            accuracies.append(np.argmin(diffs)==data['match_indices'][i])
            acc_angles.append(np.abs(data['angles'] - angles))

        epoch_accuracies.append(np.mean(accuracies))
        epoch_angles.append(np.mean(acc_angles))

    return np.mean(epoch_accuracies), np.mean(epoch_angles)

if __name__ == "__main__":
    dataset = MockData()   
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=MockData.collate_fn)

    # dataset = Semantic3dObjectData('./data/semantic3d', 'sg27_station5_intensity_rgb')
    # dataloader = DataLoader(dataset, batch_size=8, collate_fn=Semantic3dObjectData.collate_fn, shuffle=False)

    loss_dict = {}
    acc_dict = {}
    for lr in (5e-3, 1e-3, 5e-4):
        for margin in (0.2, ):
            key = (lr, margin)

            model = Net().to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.TripletMarginLoss(margin=margin)
            criterion_angle = nn.MSELoss()

            loss_dict[key] = []
            acc_dict[key] = []
            for epoch in range(25):
                print(f'Key {key} epoch {epoch} \r', end="")
                loss = train_epoch()
                loss_dict[key].append(loss)

                acc, acc_angle = test()
                acc_dict[key].append(acc)
                # print("\n",acc_angle)
            print()

plt.figure()
for k in loss_dict.keys():
    l=loss_dict[k]
    line, = plt.plot(l)
    line.set_label(k)
plt.title('Losses')
plt.gca().set_ylim(bottom=0.0) #Set the bottom to 0.0
plt.legend()

plt.figure()
for k in acc_dict.keys():
    l=acc_dict[k]
    line, = plt.plot(l)
    line.set_label(k)
plt.title('Accuracy')
plt.gca().set_ylim(bottom=0.0) #Set the bottom to 0.0
plt.legend()
plt.show()
