import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

EMBED_DIM = 32

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #self.transformer = nn.Transformer(d_model=EMBED_DIM, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=2*EMBED_DIM)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=4, dim_feedforward=2*EMBED_DIM)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 2) #TODO: Add norm?

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=EMBED_DIM, nhead=4, dim_feedforward=2*EMBED_DIM)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, 2)

    def forward(self, objects, descriptions):
        encoded = self.encoder(objects)
        decoded = self.decoder(descriptions, encoded)
        return encoded, decoded

objects_embed = nn.Embedding(4, EMBED_DIM)
with torch.no_grad():
    objects_features = objects_embed(torch.arange(4))

descriptions_embed = nn.Embedding(3, EMBED_DIM)
with torch.no_grad():
    descriptions_features = descriptions_embed(torch.arange(3))

objects_features = torch.unsqueeze(objects_features, dim=1)
descriptions_features = torch.unsqueeze(descriptions_features, dim=1)

model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.TripletMarginLoss(margin=0.2)

losses = []
for epoch in range(25):
    optimizer.zero_grad()
    
    encoded, decoded = model(objects_features, descriptions_features)
    a, p, n = torch.tensor([0,1,2]), torch.tensor([0,1,2]), torch.tensor([1,2,0])
    
    loss = criterion(encoded[a], decoded[p], encoded[n])
    loss.backward()
    optimizer.step()

    if epoch%2==0:
        losses.append(loss.cpu().detach().numpy())

for l in losses:
    print(f'{l:0.3f}')
    




