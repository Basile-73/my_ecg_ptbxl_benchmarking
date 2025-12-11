# Implementation of LSTM approach presented in
# Deep Learning Models for Denoising ECG Signals Corneliu T.C. Arsene, Richard Hankins, Hujun Yin
# https://ieeexplore.ieee.org/document/8902833
# Translated from Keras to PyTorch (source: https://github.com/Armos05/DCE-MRI-data-noise-reduction/blob/main/Deep%20learning%20Filters/Deep%20Models.py)

import torch
import torch.nn as nn
import torch.nn.functional as F

def _calc_out_len(L):
        for _ in range(3):
            L = (L + 2*1 - 2) // 4 + 1   # avg-pool formula
        return L

class CNN_Denoising(nn.Module):
    def __init__(self, squence_length):
        super().__init__()

        #self.seq_len = squence_length
        self.c1 = nn.Conv1d(1, 36, 19, padding=9)        # 512 → 512
        self.b1 = nn.BatchNorm1d(36)

        self.c2 = nn.Conv1d(36, 36, 19, padding=9)       # 128 → 128
        self.b2 = nn.BatchNorm1d(36)

        self.c3 = nn.Conv1d(36, 36, 19, padding=9)       # 32 → 32
        self.b3 = nn.BatchNorm1d(36)

        out_len = _calc_out_len(squence_length)
        self.fc = nn.Linear(36 * out_len, squence_length)                 # 288 → 512

    def forward(self, x):
        x = x.squeeze(2)
        # x: (B, 1, 512)

        x = F.relu(self.c1(x))
        x = self.b1(x)
        x = F.leaky_relu(x, 0.01)
        x = F.avg_pool1d(x, kernel_size=2, stride=4, padding=1)   # 512 → 128

        x = F.relu(self.c2(x))
        x = self.b2(x)
        x = F.leaky_relu(x, 0.01)
        x = F.avg_pool1d(x, kernel_size=2, stride=4, padding=1)   # 128 → 32

        x = F.relu(self.c3(x))
        x = self.b3(x)
        x = F.leaky_relu(x, 0.01)
        x = F.avg_pool1d(x, kernel_size=2, stride=4, padding=1)   # 32 → 8

        x = x.flatten(1)                                          # (B, 36*8 = 288)

        x = self.fc(x)                                            # (B, 512)
        x = x.unsqueeze(1)                                     # (B, 1, 512)
        x = x.unsqueeze(2)                                     # (B, 1, 1, 512)
        return x


# test forward pass
# seq_length = 10800
# model = CNN_Denoising(squence_length=seq_length)
# x = torch.randn(32,1,1,seq_length)
# y = model(x)
# print(y.shape)
