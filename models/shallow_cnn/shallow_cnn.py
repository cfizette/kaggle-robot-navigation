import math
import torch
import torch.nn.functional as F 
import torch.nn as nn

'''
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
'''

class ShallowCNN(nn.Module):
    def __init__(self, 
                 sequence_length=128,
                 n_channels=20,
                 conv1_width = 10,
                 max_pool_kernel_size=25,
                 max_pool_stride=5,
                 dropout_rate=0.5):
        '''
        sequence_length: number of measurements in each sequennce
        n_channles: number of channels for convolution layers
        max_pool_kernel_size: size along 2nd axis (not sure on this wording)
        max_pool_stride: stride along 2nd axis
        '''
        super(ShallowCNN, self).__init__()
        self.n_channels = n_channels
        self.dropout_rate = dropout_rate
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_channels, kernel_size=(1,conv1_width))
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(10,1))
        self.post_conv_width = math.floor(1 + (sequence_length - conv1_width)/self.conv1.stride[1])
        
        self.conv2_bn = nn.BatchNorm2d(n_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, max_pool_kernel_size), stride=(1, max_pool_stride))
        max_pool_w_out = math.floor(1 + (self.post_conv_width - max_pool_kernel_size)/max_pool_stride)
        
        self.linear = nn.Linear(in_features=n_channels*max_pool_w_out, out_features=9)
        
    def forward(self, x):
        x = self.conv1(x)
        #x = F.dropout2d(x, p=0.5)
        
        x = self.conv2(x)
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv2_bn(x))
        #x = F.relu(self.conv2_bn(x))
        
        x = x.view((-1, self.n_channels, self.post_conv_width))
        x = self.max_pool(x)
        
        x = x.view((x.size(0), -1))
        x = self.linear(x)
        
        x = F.log_softmax(x, dim=1)
        return x
