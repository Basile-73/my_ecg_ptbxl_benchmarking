import torch.nn as nn
import torch

try:
    from mamba_ssm import Mamba
except Exception as e:
    raise ImportError("Install mamba-ssm: pip install mamba-ssm")

class ResidualMambaLayer(nn.Module):
    def __init__(self, channels, d_state=256, d_conv=4, expand=4, bidirectional=False):
        super().__init__()
        self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        if bidirectional:
            self.mamba_backward = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
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

from mamba_ssm.models.mixer_seq_simple import create_block

class ResidualMambaBlockLayer(nn.Module):
    def __init__(self, channels, d_state, d_conv, expand, d_intermediate, mamba_type, headdim):
        super().__init__()
        ssm_cfg = dict(d_state=d_state, d_conv=d_conv, expand=expand)
        ssm_cfg['layer'] = mamba_type
        if mamba_type == 'Mamba2':
            ssm_cfg['headdim'] = headdim
            ssm_cfg['use_mem_eff_path'] = False
            ssm_cfg['chunk_size'] = 64
        self.block = create_block(
            d_model=channels,
            d_intermediate=d_intermediate,
            ssm_cfg=ssm_cfg,
            layer_idx=0
        )

    def forward(self, x):                  # (B,C,1,W)
        x = x.squeeze(2).transpose(1,2)    # (B,W,C)
        h, r = self.block(x, None)
        y = h + r
        return y.transpose(1,2).unsqueeze(2)
