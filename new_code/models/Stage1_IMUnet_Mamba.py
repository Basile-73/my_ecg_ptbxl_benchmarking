# -*- coding: utf-8 -*-
"""
Variable-Length IMUnet with Mamba-Enhanced Bottleneck for ECG Denoising
code name: Stage1_2_IMUnet_mamba_merge_bn_big_varlen_upconv.py

Created on Sat Aug 15 18:15:03 2020
@author: Lishen Qiu

Modified by: Basile Morel
On line 207: Replace the merge operation in the bottleneck with Mamba.
Mamba's selective state-space modeling captures long-range contextual
differences more effectively than local convolutions.

Modified for variable-length support: Dynamically calculates upsample sizes
based on input_length to support training on short sequences and evaluating
on longer sequences without interpolation wrappers.
"""

from __future__ import division
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

from einops import rearrange

try:
    from mamba_ssm import Mamba
except Exception as e:
    raise ImportError("Install mamba-ssm: pip install mamba-ssm")

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        hidden_channels = in_channels // ratio
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        avg_out = self.shared_mlp(F.adaptive_avg_pool2d(x, 1))  # (B, C, 1, 1)
        max_out = self.shared_mlp(F.adaptive_max_pool2d(x, 1))  # (B, C, 1, 1)
        out = avg_out + max_out                                # (B, C, 1, 1)
        return self.sigmoid(out)                               # (B, C, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size_L, kernel_size_W, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=(kernel_size_L, kernel_size_W),
            stride=stride,
            padding=(0, kernel_size_W // 2),
            bias=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        avg_map = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_map, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        combined = torch.cat([avg_map, max_map], dim=1)  # (B, 2, H, W)
        out = self.conv(combined)                        # (B, 1, H, W)
        return self.sigmoid(out)                         # (B, 1, H, W)


class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
#        print(self.avg_pool.size())
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MambaMerge(nn.Module):
    """
    Mamba-based merge module that processes concatenated features through
    selective state-space modeling for effective context fusion.

    This module is inherently sequence-agnostic and handles variable lengths:
    - Uses einops.rearrange to convert 2D features to sequences
    - Mamba processes sequences of any length
    - Reshapes back to 2D using stored dimensions
    """
    def __init__(self, in_channels, out_channels, d_state=64, expand=2):
        super().__init__()
        # 1x1 convolution to project concatenated features to target channels
        self.input_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # Layer normalization for stable training
        self.norm = nn.LayerNorm(out_channels)

        # Mamba block for sequence modeling
        self.mamba = Mamba(d_model=out_channels, d_state=d_state, expand=expand)

    def forward(self, x):
        # x: (B, in_channels, H, W) - typically (B, out_ch*3, H, W)
        B, C_in, H, W = x.shape

        # Project to target number of channels
        x = self.input_proj(x)  # (B, out_channels, H, W)
        _, C_out, _, _ = x.shape

        # Reshape from 2D to sequence: (B, C, H, W) -> (B, H*W, C)
        x_seq = rearrange(x, 'b c h w -> b (h w) c')

        # Apply layer normalization
        x_norm = self.norm(x_seq)  # (B, H*W, C)

        # Apply Mamba with residual connection
        x_mamba = self.mamba(x_norm) + x_seq  # (B, H*W, C)

        # Reshape back to 2D: (B, H*W, C) -> (B, C, H, W)
        out = rearrange(x_mamba, 'b (h w) c -> b c h w', h=H, w=W)

        return out  # (B, out_channels, H, W)


class conv_1_block_DW(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch,kernel_size_L,kernel_size_W,stride):
        super(conv_1_block_DW, self).__init__()

        self.conv = nn.Sequential(


            nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
#            SELayer(out_ch, 8),

            )
        self.ca =ChannelAttention(out_ch,8)
        self.sp =SpatialAttention(kernel_size_L,kernel_size_W,stride=1)
    def forward(self, x):
        x = self.conv(x)
        x = self.ca(x)* x
#        x = self.sp(x)* x
        return x

class conv_1_block_MD(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch,kernel_size_L,kernel_size_W,stride):
        super(conv_1_block_MD, self).__init__()

        self.conv = nn.Sequential(


            nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
#            SELayer(out_ch, 8),

            )
        self.ca =ChannelAttention(out_ch,8)
        self.sp =SpatialAttention(kernel_size_L,kernel_size_W,stride=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.ca(x)* x
#        x = self.sp(x)* x
        return x

class conv_1_block_UP(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch,kernel_size_L,kernel_size_W,stride):
        super(conv_1_block_UP, self).__init__()

        self.conv = nn.Sequential(


            nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
#            SELayer(out_ch, 8),
            )
        self.ca =ChannelAttention(out_ch,8)
        self.sp =SpatialAttention(kernel_size_L,kernel_size_W,stride=1)
    def forward(self, x):
        x = self.conv(x)
        x = self.ca(x)* x
#        x = self.sp(x)* x
        return x

class Context_comparison(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size_L, kernel_size_W, stride=1):
        super().__init__()

        # Two parallel convolutions: one standard, one dilated
        self.conv_local = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (kernel_size_L, kernel_size_W),
                      stride=stride, padding=(0, kernel_size_W // 2), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv_dilated = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (kernel_size_L, kernel_size_W),
                      stride=stride, padding=(0, (kernel_size_W + 7) // 2),
                      bias=True, dilation=5),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # Channel attention modules for both branches
        self.ca = ChannelAttention(out_ch, 8)

        # Merge output channels from x1, x2, and (x1 - x2) using Mamba
        self.merge = MambaMerge(in_channels=out_ch * 3, out_channels=out_ch, d_state=256, expand=4)

    def forward(self, x):
        # x: (B, in_ch, H, W)

        # Local context branch
        x1 = self.conv_local(x)              # (B, out_ch, H, W)
        x1 = self.ca(x1) * x1
        x1 = self.conv_local(x1)             # (B, out_ch, H, W)
        x1 = self.ca(x1) * x1

        # Dilated context branch
        x2 = self.conv_dilated(x)            # (B, out_ch, H, W)
        x2 = self.ca(x2) * x2
        x2 = self.conv_dilated(x2)           # (B, out_ch, H, W)
        x2 = self.ca(x2) * x2

        # Context difference
        x3 = x1 - x2                         # (B, out_ch, H, W)

        # Concatenate and fuse
        combined = torch.cat((x1, x2, x3), dim=1)  # (B, out_ch*3, H, W)
        out = self.merge(combined)                 # (B, out_ch, H, W)

        return out

class IMUnet(nn.Module):
    """
    IMUnet with Mamba-enhanced bottleneck and variable-length support.

    This version uses MambaMerge in the bottleneck for better context fusion
    and dynamically calculates upsample sizes based on input_length to support
    training on short sequences and evaluating on longer sequences.

    Args:
        in_channels (int): Number of input channels (default: 1)
        input_length (int): Expected input sequence length in samples (default: 3600)

    Architecture:
        - Downsampling pattern: AvgPool strides 5×2×2 = 10x total reduction
        - Bottleneck: Context_comparison with MambaMerge for selective state-space modeling
        - Upsample sizes calculated dynamically in forward() using F.interpolate
        - MambaMerge handles variable sequence lengths natively through einops.rearrange
    """
    def __init__(self, in_channels=1, input_length=3600):
        super(IMUnet, self).__init__()

        # Validate input_length to prevent collapse after pooling
        # Minimum length based on downsampling pattern: 5×2×2 = 10x total
        # Need at least 20 samples to avoid zero-sized tensors after pooling
        min_length = 20
        if input_length < min_length:
            raise ValueError(
                f"input_length must be at least {min_length} samples. "
                f"Got {input_length}. The model uses pooling strides of 5×2×2 = 10x total reduction, "
                f"which would collapse inputs shorter than {min_length} to zero."
            )

        # Store input length and input channels for reference
        self.input_length = input_length
        self.in_channels = in_channels

        # Use in_channels parameter instead of hardcoded 1
        self.conv1_1=conv_1_block_DW(in_channels, 16, kernel_size_L=1,kernel_size_W=25,stride=1)
        self.conv1_2=conv_1_block_DW(16, 16, kernel_size_L=1,kernel_size_W=25,stride=1)
        self.conv1_3=conv_1_block_DW(16, 16, kernel_size_L=1,kernel_size_W=25,stride=1)

        self.conv2_1=conv_1_block_DW(16, 32, kernel_size_L=1,kernel_size_W=15,stride=1)
        self.conv2_2=conv_1_block_DW(32, 32, kernel_size_L=1,kernel_size_W=15,stride=1)
        self.conv2_3=conv_1_block_DW(32, 32, kernel_size_L=1,kernel_size_W=15,stride=1)

        self.conv3_1=conv_1_block_DW(32, 48, kernel_size_L=1,kernel_size_W=5,stride=1)
        self.conv3_2=conv_1_block_DW(48, 48, kernel_size_L=1,kernel_size_W=5,stride=1)
        self.conv3_3=conv_1_block_DW(48, 48, kernel_size_L=1,kernel_size_W=5,stride=1)

        self.conv4_1=conv_1_block_MD(48, 64, kernel_size_L=1,kernel_size_W=3,stride=1)
#        self.conv4_2=conv_1_block_MD(64, 64, kernel_size_L=1,kernel_size_W=3,stride=1)
        self.conv4_2=Context_comparison(64, 64, kernel_size_L=1,kernel_size_W=3,stride=1)
        self.conv4_3=conv_1_block_MD(64, 64, kernel_size_L=1,kernel_size_W=3,stride=1)

        self.conv5_1=conv_1_block_UP(48+64, 48, kernel_size_L=1,kernel_size_W=5,stride=1)
        self.conv5_2=conv_1_block_UP(48, 48, kernel_size_L=1,kernel_size_W=5,stride=1)
        self.conv5_3=conv_1_block_UP(48, 32, kernel_size_L=1,kernel_size_W=5,stride=1)

        self.conv6_1=conv_1_block_UP(32+32, 32, kernel_size_L=1,kernel_size_W=15,stride=1)
        self.conv6_2=conv_1_block_UP(32, 32, kernel_size_L=1,kernel_size_W=15,stride=1)
        self.conv6_3=conv_1_block_UP(32, 16, kernel_size_L=1,kernel_size_W=15,stride=1)

        self.conv7_1=conv_1_block_UP(16+16, 16, kernel_size_L=1,kernel_size_W=25,stride=1)
        self.conv7_2=conv_1_block_UP(16, 16, kernel_size_L=1,kernel_size_W=25,stride=1)
        self.conv7_3=conv_1_block_UP(16, 16, kernel_size_L=1,kernel_size_W=25,stride=1)

        self.conv1m1 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1,1),padding=0)

        # Downsampling layers (strides: 5, 2, 2)
        self.avepool1 = nn.AvgPool2d((1, 5), stride=5)
        self.avepool2 = nn.AvgPool2d((1, 2), stride=2)
        self.avepool3 = nn.AvgPool2d((1, 2), stride=2)

        # Upsampling layers using transpose convolutions (matching downsampling strides)
        self.upsample1 = nn.ConvTranspose2d(64, 64, kernel_size=(1, 2), stride=(1, 2))  # 2x upsample
        self.upsample2 = nn.ConvTranspose2d(32, 32, kernel_size=(1, 2), stride=(1, 2))  # 2x upsample
        self.upsample3 = nn.ConvTranspose2d(16, 16, kernel_size=(1, 5), stride=(1, 5))  # 5x upsample

    def forward(self, x):
        # Validate input shape matches expected input_length
        actual_length = x.shape[-1]
        if actual_length != self.input_length:
            raise RuntimeError(
                f"Input length mismatch: model was initialized with input_length={self.input_length}, "
                f"but received input with length={actual_length}. "
                f"Please instantiate a new model with the correct input_length."
            )

        # Validate input is multiple of 20 for proper upsampling
        if actual_length % 20 != 0:
            raise ValueError(
                f"Input length must be a multiple of 20 for transpose convolution upsampling. "
                f"Got length={actual_length}."
            )

        # Encoder
        x1 = self.conv1_3(self.conv1_2(self.conv1_1(x)))        # (B, 16, 1, L)
        x1_pooled = self.avepool1(x1)                           # (B, 16, 1, L/5)

        x2 = self.conv2_3(self.conv2_2(self.conv2_1(x1_pooled)))# (B, 32, 1, L/5)
        x2_pooled = self.avepool2(x2)                           # (B, 32, 1, L/10)

        x3 = self.conv3_3(self.conv3_2(self.conv3_1(x2_pooled)))# (B, 48, 1, L/10)
        x3_pooled = self.avepool3(x3)                           # (B, 48, 1, L/20)

        # Bottleneck with Mamba-enhanced context comparison
        x4 = self.conv4_3(self.conv4_2(self.conv4_1(x3_pooled)))# (B, 64, 1, L/20)

        # Decoder stage 1 - Transpose convolution 2x upsample
        x4_up = self.upsample1(x4)                              # (B, 64, 1, L/10)
        x5 = torch.cat((x4_up, x3), dim=1)                      # (B, 112, 1, L/10)
        x5 = self.conv5_3(self.conv5_2(self.conv5_1(x5).add(x3)).add(x3))
        # x5: (B, 32, 1, L/10)

        # Decoder stage 2 - Transpose convolution 2x upsample
        x5_up = self.upsample2(x5)                              # (B, 32, 1, L/5)
        x6 = torch.cat((x5_up, x2), dim=1)                      # (B, 64, 1, L/5)
        x6 = self.conv6_3(self.conv6_2(self.conv6_1(x6).add(x2)).add(x2))
        # x6: (B, 16, 1, L/5)

        # Decoder stage 3 - Transpose convolution 5x upsample
        x6_up = self.upsample3(x6)                              # (B, 16, 1, L)
        x7 = torch.cat((x6_up, x1), dim=1)                      # (B, 32, 1, L)
        x7 = self.conv7_3(self.conv7_2(self.conv7_1(x7).add(x1)).add(x1))
        # x7: (B, 16, 1, L)

        # Output
        out = self.conv1m1(x7)                                  # (B, 1, 1, L)
        return out



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Demonstrate with default 3600 samples
    model = IMUnet(input_length=3600).to(device)
    print("Model with input_length=3600:")
    summary(model, (1,1,3600))

    # Example: model = IMUnet(input_length=5000).to(device)
