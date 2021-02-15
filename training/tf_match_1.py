import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.TripletMarginLoss(margin=0.2)

loss_dict = {}

for lr in (1e-3, 5e-4, 1e-4):
    for epoch in range(25):
        optimizer.zero_grad()
        
        encoded, decoded = model(objects_features, descriptions_features)
        a, p, n = torch.tensor([0,1,2]), torch.tensor([0,1,2]), torch.tensor([1,2,0])
        
        loss = criterion(encoded[a], decoded[p], encoded[n])
        loss.backward()
        optimizer.step()

        if epoch%2==0:
            losses.append(loss.cpu().detach().numpy())

plt.figure()
for k in loss_dict.keys():
    l=loss_dict[k]
    line, = plt.plot(l)
    line.set_label(k)
plt.title('Losses')
plt.gca().set_ylim(bottom=0.0) #Set the bottom to 0.0
plt.legend()
