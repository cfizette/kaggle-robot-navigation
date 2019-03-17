import torch
import torch.functional as F 
import torch.nn as nn


class ShallowCNN(nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(1,10))
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(10,1))
        self.conv2_bn = nn.BatchNorm2d(20)
        self.max_pool = nn.MaxPool2d(kernel_size=(1,25), stride=(1,5))
        self.linear = nn.Linear(in_features=20*19, out_features=9)
        
    def forward(self, x):
        x = self.conv1(x)
        #x = F.dropout2d(x, p=0.5)
        
        x = self.conv2(x)
        x = F.dropout2d(x, p=0.5)
        x = F.elu(self.conv2_bn(x))
        
        x = x.view((-1,20,119))
        x = self.max_pool(x)
        
        x = x.view((x.size(0), -1))
        x = self.linear(x)
        
        x = F.log_softmax(x, dim=1)
        return x