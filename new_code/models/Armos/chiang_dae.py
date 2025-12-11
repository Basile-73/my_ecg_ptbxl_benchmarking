# Implementation of FCN_DAE approach presented in
# Chiang, H. T., Hsieh, Y. Y., Fu, S. W., Hung, K. H., Tsao, Y., & Chien, S. Y. (2019).
# Noise reduction in ECG signals using fully convolutional denoising autoencoders.
# IEEE Access, 7, 60806-60813.
# Translated from Keras to PyTorch (source: https://github.com/Armos05/DCE-MRI-data-noise-reduction/blob/main/Deep%20learning%20Filters/Deep%20Models.py)

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN_DAE(nn.Module):
    def __init__(self):
        super().__init__()

        # ----- Encoder -----
        self.c1 = nn.Conv1d(1, 40, 16, stride=2, padding=8)
        self.b1 = nn.BatchNorm1d(40)

        self.c2 = nn.Conv1d(40, 20, 16, stride=2, padding=8)
        self.b2 = nn.BatchNorm1d(20)

        self.c3 = nn.Conv1d(20, 20, 16, stride=2, padding=8)
        self.b3 = nn.BatchNorm1d(20)

        self.c4 = nn.Conv1d(20, 20, 16, stride=2, padding=8)
        self.b4 = nn.BatchNorm1d(20)

        self.c5 = nn.Conv1d(20, 40, 16, stride=2, padding=8)
        self.b5 = nn.BatchNorm1d(40)

        self.c6 = nn.Conv1d(40, 1, 16, stride=1, padding=8)
        self.b6 = nn.BatchNorm1d(1)

        # ----- Decoder -----
        self.t1 = nn.ConvTranspose1d(1, 1, 16, stride=1, padding=8)
        self.tb1 = nn.BatchNorm1d(1)

        self.t2 = nn.ConvTranspose1d(1, 40, 16, stride=2, padding=8, output_padding=1)
        self.tb2 = nn.BatchNorm1d(40)

        self.t3 = nn.ConvTranspose1d(40, 20, 16, stride=2, padding=8, output_padding=1)
        self.tb3 = nn.BatchNorm1d(20)

        self.t4 = nn.ConvTranspose1d(20, 20, 16, stride=2, padding=8, output_padding=1)
        self.tb4 = nn.BatchNorm1d(20)

        self.t5 = nn.ConvTranspose1d(20, 20, 16, stride=2, padding=8, output_padding=1)
        self.tb5 = nn.BatchNorm1d(20)

        self.t6 = nn.ConvTranspose1d(20, 40, 16, stride=2, padding=8, output_padding=1)
        self.tb6 = nn.BatchNorm1d(40)

        self.out = nn.ConvTranspose1d(40, 1, 16, stride=1, padding=8)

    def forward(self, x):
        x = x.squeeze(2)
        # ----- Encoder -----
        x = F.elu(self.b1(self.c1(x)))   # 512 → 256
        x = F.elu(self.b2(self.c2(x)))   # 256 → 128
        x = F.elu(self.b3(self.c3(x)))   # 128 → 64
        x = F.elu(self.b4(self.c4(x)))   # 64  → 32
        x = F.elu(self.b5(self.c5(x)))   # 32  → 16
        x = F.elu(self.b6(self.c6(x)))   # 16  → 16

        # ----- Decoder -----
        x = F.elu(self.tb1(self.t1(x)))  # 16 → 16
        x = F.elu(self.tb2(self.t2(x)))  # 16 → 32
        x = F.elu(self.tb3(self.t3(x)))  # 32 → 64
        x = F.elu(self.tb4(self.t4(x)))  # 64 → 128
        x = F.elu(self.tb5(self.t5(x)))  # 128 → 256
        x = F.elu(self.tb6(self.t6(x)))  # 256 → 512
        x = self.out(x)                  # 512 → 512

        x = x.unsqueeze(2)
        return x

# test forward pass
# from length_wrapper import AnyLengthWrapper
# model = AnyLengthWrapper(FCN_DAE(), factor=32)
# x = torch.randn(32,1,1,10800)
# y = model(x)
# print(y.shape)
