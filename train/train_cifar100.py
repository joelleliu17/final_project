import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import os
import pickle
import numpy as np
from torch.utils.data import Dataset,DataLoader

from ../model.models import *

trainset = torchvision.datasets.CIFAR100(root='../dataset', train=True, download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root='../dataset', train=False, download=False, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# --- Hyperparameters ---

batch_size = 128
num_epochs = 100
base_lr = 0.001

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = resnet50().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=base_lr)


# --- Training & Evaluation Loop ---

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    total, correct = 0, 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {running_loss/len(trainloader):.4f} Acc: {100.*correct/total:.2f}%")



## 存模型
torch.save(model, 'model.pth')

## 存權重
torch.save(model.state_dict(), 'model_weights.pth')
