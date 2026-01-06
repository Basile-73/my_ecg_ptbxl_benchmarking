# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 18:15:03 2020

@author: Lishen Qiu
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from numpy.linalg import svd
from numpy.random import normal
import math
from math import sqrt
from torchsummary import summary
import scipy.io as io
import numpy as np
import torch.optim as optim
import torch.utils.data
import torch
import os
import os.path as osp
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from mamba_layer import ResidualMambaLayer

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size_L,kernel_size_W,stride):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class conv_1_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch,kernel_size_L,kernel_size_W,stride):
        super(conv_1_block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            )

    def forward(self, x):
        x1 = self.conv1(x)
        return x1

class conv_2_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch,kernel_size_L,kernel_size_W,stride):
        super(conv_2_block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.ReLU(inplace=True),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.ReLU(inplace=True),
            )
        self.ca =ChannelAttention(out_ch,8)
        self.sp =SpatialAttention(kernel_size_L,kernel_size_W,stride=1)
    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.ca(x1)* x1
        x1 = self.conv2(x1)
        x1 = self.ca(x1)* x1

        xout=x1+x
        return xout




class MambaDRnet(nn.Module):#库中的torch.nn.Module模块
    def __init__(self,in_channels =1, bidirectional=False):
        super(MambaDRnet, self).__init__()


        # self.conv1_1=conv_1_block( 2, 32, kernel_size_L=1,kernel_size_W=3, stride=1)
        # self.mamba_1=ResidualMambaLayer(32, bidirectional=bidirectional)


        self.conv2_1=conv_1_block( 2, 32, kernel_size_L=1,kernel_size_W=5, stride=1)
        self.mamba_2=ResidualMambaLayer(32, bidirectional=bidirectional)


        # self.conv3_1=conv_1_block( 2, 32, kernel_size_L=1,kernel_size_W=13, stride=1)
        # self.mamba_3=ResidualMambaLayer(32, bidirectional=bidirectional)


        self.conv4_1=conv_1_block( 2, 32, kernel_size_L=1,kernel_size_W=15, stride=1)
        self.mamba_4=ResidualMambaLayer(32, bidirectional=bidirectional)


        # self.conv1m1_1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1,3),padding=(0,1))
        self.conv1m1_2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1,5),padding=(0,2))
        # self.conv1m1_3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1,13),padding=(0,6))
        self.conv1m1_4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1,15),padding=(0,7))

    def forward(self, x):

        # x1 = self.conv1_1(x)
        # x1 = self.mamba_1(x1)
        # x1 = self.conv1m1_1(x1)

        x2 = self.conv2_1(x)
        x2 = self.mamba_2(x2)
        x2 = self.conv1m1_2(x2)

        # x3 = self.conv3_1(x)
        # x3 = self.mamba_3(x3)
        # x3 = self.conv1m1_3(x3)

        x4 = self.conv4_1(x)
        x4 = self.mamba_4(x4)
        x4 = self.conv1m1_4(x4)

        # Xout=x1+x2+x3+x4
        Xout=x2+x4

        return Xout

# from torchsummary import summary
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = MambaDRnet(in_channels =1).to(device)
# summary(model, (2,1,3600))
