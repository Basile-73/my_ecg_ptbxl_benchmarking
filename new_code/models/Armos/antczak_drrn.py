# Implementation of DRNN approach presented in
# Antczak, K. (2018). Deep recurrent neural networks for ECG signal denoising.
# arXiv preprint arXiv:1807.11551.
# Translated from Keras to PyTorch (source: https://github.com/Armos05/DCE-MRI-data-noise-reduction/blob/main/Deep%20learning%20Filters/Deep%20Models.py)


import torch
import torch.nn as nn

class DRRN_Denoising(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            batch_first=True,
            num_layers=1,
            bidirectional=False
        )

        self.d1 = nn.Linear(64, 64)
        self.d2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

        self.act = nn.ReLU()

    def forward(self, x):
        x = x.squeeze(2)
        x = x.permute(0, 2, 1) # ! different

        # x: (B, 512, 1)
        x, _ = self.lstm(x)         # (B, 512, 64)

        x = self.act(self.d1(x))    # (B, 512, 64)
        x = self.act(self.d2(x))    # (B, 512, 64)

        x = self.out(x)             # (B, 512, 1)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(2)        # (B, 1, 1, 512)
        return x

# test forward pass
# model = DRRN_Denoising()
# x = torch.randn(32,1,1,10800)
# y = model(x)
# print(y.shape)  # should be (3,1,512)
