import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import os
import pickle
import numpy as np

from ../model.models import *
 
from torch.utils.data import Dataset,DataLoader
 
class CIFAR100Dataset(Dataset):
    def __init__(self, path, transform=None, train=False):
        if train:
            sub_path = 'train'
        else:
            sub_path = 'test'
        with open(os.path.join(path, sub_path), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform
 
    def __len__(self):
        return len(self.data['fine_labels'.encode()])
 
    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))
 
        if self.transform:
            image = self.transform(image)
        return image, label

# 訓練與測試資料的轉換
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
])


## DataLoader
trainset = torchvision.datasets.CIFAR100(root='./dataset/TTA', train=True, download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root='./dataset/TTA', train=False, download=False, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# --- Hyperparameters ---
batch_size = 128
num_epochs = 100
base_lr = 0.001

## 載入模型
model = resnet50().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=base_lr)

## 訓練
for epoch in range(num_epochs):
    model.train()
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
