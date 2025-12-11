import torch.nn as nn
import torch

try:
    from mamba_ssm import Mamba
except Exception as e:
    raise ImportError("Install mamba-ssm: pip install mamba-ssm")

class ResidualMambaLayer(nn.Module):
    def __init__(self, channels, d_state=256, expand=4, bidirectional=False):
        super().__init__()
        self.mamba = Mamba(d_model=channels, d_state=d_state, expand=expand)
        if bidirectional:
            self.mamba_backward = Mamba(d_model=channels, d_state=d_state, expand=expand)
        else:
            self.mamba_backward = None


    def forward(self, x):          # x: (B, C, 1, W)
        b, c, h, w = x.shape
        x_seq = x.squeeze(2).transpose(1, 2)      # (B, W, C)
        y = self.mamba(x_seq)                     # (B, W, C)
        if self.mamba_backward is not None:
            x_seq_rev = torch.flip(x_seq, dims=[1])   # (B, W, C)
            y_rev = self.mamba_backward(x_seq_rev)     # (B, W, C)
            y_rev = torch.flip(y_rev, dims=[1])        # (B, W, C)
            y = (y + y_rev) / 2
        y = y.transpose(1, 2).unsqueeze(2)        # (B, C, 1, W)
        return x + y
