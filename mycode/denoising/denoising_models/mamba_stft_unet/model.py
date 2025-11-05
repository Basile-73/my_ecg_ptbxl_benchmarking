# model_mamba_ecg_stft.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mamba_ssm import Mamba
except Exception as e:
    raise ImportError("Install mamba-ssm: pip install mamba-ssm")

class STFTWrapper(nn.Module):
    def __init__(self, n_fft=64, hop_length=8, win_length=64):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
    def forward(self, x):
        # x: (B, 1, T)
        X = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True
        )  # (B, F, Tspec)
        return X
    def inverse(self, X, length):
        y = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=length
        )
        return y.unsqueeze(1)

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, stride=s, padding=p),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, k, padding=1),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
        )
    def forward(self, x):
        return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2)
        self.conv = ConvBlock(c_out + c_skip, c_out)
    def forward(self, x, skip):
        x = self.up(x)
        if x.size(-1) != skip.size(-1):
            x = F.pad(x, (0, skip.size(-1)-x.size(-1)))
        if x.size(-2) != skip.size(-2):
            x = F.pad(x, (0,0,0, skip.size(-2)-x.size(-2)))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64, expand=2):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        # x: (B, C, F, T) where C=d_model
        # Rearrange to treat F*T as sequence length, C as features
        B, C, F, T = x.shape
        x_flat = rearrange(x, 'b c f t -> b (f t) c')
        y = self.mamba(self.norm(x_flat)) + x_flat
        return rearrange(y, 'b (f t) c -> b c f t', f=F, t=T)

class TinyMambaSTFTUNet(nn.Module):
    def __init__(self, base_ch=32, mamba_dim=64, mamba_depth=3):
        super().__init__()
        self.stft = STFTWrapper(n_fft=64, hop_length=8, win_length=64)

        self.enc1 = ConvBlock(2, base_ch)       # real+imag channels
        self.enc2 = ConvBlock(base_ch, base_ch*2, s=2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4, s=2)

        self.bridge = nn.Conv2d(base_ch*4, mamba_dim, 1)
        self.mambas = nn.ModuleList([MambaBlock(mamba_dim) for _ in range(mamba_depth)])
        self.unbridge = nn.Conv2d(mamba_dim, base_ch*4, 1)

        self.up2 = UpBlock(base_ch*4, base_ch*2, base_ch*2)
        self.up1 = UpBlock(base_ch*2, base_ch, base_ch)

        self.head = nn.Conv2d(base_ch, 2, 1)

    def forward(self, x):
        # Handle 4D input (B, 1, 1, T) - convert to 3D for STFT processing
        B, _, _, T = x.shape
        x = x.squeeze(2)  # (B, 1, 1, T) -> (B, 1, T)
        X = self.stft(x)  # (B, F, Tspec)
        X_in = torch.stack([X.real, X.imag], dim=1)  # (B, 2, F, Tspec)

        s1 = self.enc1(X_in)
        s2 = self.enc2(s1)
        h = self.enc3(s2)

        h = self.bridge(h)
        for blk in self.mambas:
            h = blk(h)
        h = self.unbridge(h)

        h = self.up2(h, s2)
        h = self.up1(h, s1)
        Y = self.head(h)

        # Masking approach: apply estimated mask on input spectrum
        mask = torch.tanh(Y)
        X_denoised = (X_in * mask)[:,0] + 1j*(X_in * mask)[:,1]
        y = self.stft.inverse(X_denoised, length=T)
        y = y.unsqueeze(2)  # (B, 1, T) -> (B, 1, 1, T)
        return y
