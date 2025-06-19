import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from KAN import KAN_Convolutional_Layer
import mentor

# model = models.resnet50(pretrained=False)  
def resnet50():
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(2048, 100)  # CIFAR-100 有 100 類
    model = model.cuda()
    return model


class KANC_MLP_Big(nn.Module):
    def __init__(self):
        super(KANC_MLP_Big, self).__init__()
        self.kanconv1 = KAN_Convolutional_Layer(in_channels=1, out_channels=5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.kanconv2 = KAN_Convolutional_Layer(in_channels=5, out_channels=10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(10 * 8 * 8, 300)  # CIFAR10/CIFAR100輸入時需調整
        self.fc2 = nn.Linear(300, 100)

    def forward(self, x):
        x = self.kanconv1(x)
        x = self.pool(x)
        x = self.kanconv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
