import torch.nn as nn
import torch.nn.functional as F

class AnyLengthWrapper(nn.Module):
    def __init__(self, base, factor):
        super().__init__()
        self.base = base
        self.factor = factor

    def forward(self, x):
        # x is already (B,1,1,L)
        B, C1, C2, L = x.shape

        # ----- compute padded length -----
        factor = self.factor
        pad_len = (factor - (L % factor)) % factor

        if pad_len > 0:
            x = F.pad(x, (0, pad_len))   # pad length dimension ONLY

        # ----- run model -----
        y = self.base(x)   # model expects (B,1,1,L_pad)

        # ----- crop back to original length -----
        y = y[..., :L]     # keep (B,1,1,L)

        return y
