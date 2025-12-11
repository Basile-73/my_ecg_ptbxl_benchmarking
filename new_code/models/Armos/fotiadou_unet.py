# Implementation of CNN autoencoder for denoising approach presented in
# End-to-End Trained CNN Encoder-Decoder Network for Fetal ECG Signal Denoising
# https://iopscience.iop.org/article/10.1088/1361-6579/ab69b9/meta
# https://github.com/rshnn/xray-denoising
# Translated from Keras to PyTorch (source: https://github.com/Armos05/DCE-MRI-data-noise-reduction/blob/main/Deep%20learning%20Filters/Deep%20Models.py)

import torch
import torch.nn as nn
import torch.nn.functional as F

act = nn.LeakyReLU(0.01)

class CNN_encoder_decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- Encoder ----
        self.e1 = nn.Conv1d(1,   64, 15, padding=7)          # 512 → 512
        self.e2 = nn.Conv1d(1,  128, 15, stride=2, padding=7) # 512 → 256
        self.e3 = nn.Conv1d(128,256,15, stride=2, padding=7)  # 256 → 128
        self.e4 = nn.Conv1d(256,256,15, stride=2, padding=7)  # 128 → 64
        self.e5 = nn.Conv1d(256,512,15, stride=2, padding=7)  # 64  → 32
        self.e6 = nn.Conv1d(512,512,15, stride=2, padding=7)  # 32  → 16
        self.e7 = nn.Conv1d(512,1024,15,stride=2, padding=7)  # 16  → 8
        self.e8 = nn.Conv1d(1024,2048,15,stride=2, padding=7) # 8   → 4

        # ---- Decoder ----
        self.d1 = nn.ConvTranspose1d(2048,2048,15,stride=1,padding=7)               # 4 → 4
        self.d2 = nn.ConvTranspose1d(2048,1024,15,stride=2,padding=7,output_padding=1) # 4 → 8
        self.d3 = nn.ConvTranspose1d(1024,512,15,stride=2,padding=7,output_padding=1)  # 8 → 16
        self.d4 = nn.ConvTranspose1d(512,512,15,stride=2,padding=7,output_padding=1)   # 16 → 32
        self.d5 = nn.ConvTranspose1d(512,256,15,stride=2,padding=7,output_padding=1)   # 32 → 64
        self.d6 = nn.ConvTranspose1d(256,256,15,stride=2,padding=7,output_padding=1)   # 64 → 128
        self.d7 = nn.ConvTranspose1d(256,128,15,stride=2,padding=7,output_padding=1)   # 128 → 256
        self.d8 = nn.ConvTranspose1d(128,64, 15,stride=2,padding=7,output_padding=1)   # 256 → 512

        self.out = nn.ConvTranspose1d(64,1,15,stride=1,padding=7)  # 512 → 512

    def forward(self, x):
        x = x.squeeze(2)
        # Encoder + skip connections
        s1 = act(self.e1(x))            # 512×64
        s2 = act(self.e2(x))            # 256×128
        x  = act(self.e3(s2)); s3 = x   # 128×256
        x  = act(self.e4(x));  s4 = x   # 64×256
        x  = act(self.e5(x));  s5 = x   # 32×512
        x  = act(self.e6(x));  s6 = x   # 16×512
        x  = act(self.e7(x));  s7 = x   # 8×1024
        x  = act(self.e8(x))            # 4×2048

        # Decoder + skip connections
        x = act(self.d1(x))             # 4×2048
        x = act(self.d2(x)) + s7        # 8×1024
        x = act(self.d3(x)) + s6        # 16×512
        x = act(self.d4(x)) + s5        # 32×512
        x = act(self.d5(x)) + s4        # 64×256
        x = act(self.d6(x)) + s3        # 128×256
        x = act(self.d7(x)) + s2        # 256×128
        x = act(self.d8(x)) + s1        # 512×64
        x = self.out(x)                 # 512×1

        x = x.unsqueeze(2)
        return x

# # test forward pass
# from length_wrapper import AnyLengthWrapper
# model = AnyLengthWrapper(CNN_encoder_decoder(), factor=256)
# x = torch.randn(32,1,1,10800)
# y = model(x)
# print(y.shape)
