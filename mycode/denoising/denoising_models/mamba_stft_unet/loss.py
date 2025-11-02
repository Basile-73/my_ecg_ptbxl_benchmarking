# losses_ecg.py
import torch
import torch.nn as nn
import torch.fft as fft

class L1STFTBandpassLoss(nn.Module):
    def __init__(self, sr=250, w_l1=1.0, w_stft=0.5, w_bp=0.05):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.w_l1, self.w_stft, self.w_bp = w_l1, w_stft, w_bp
        self.sr = sr
    def stft_mag(self, x):
        X = torch.stft(x, n_fft=256, hop_length=64, win_length=256, return_complex=True)
        return (X.abs() + 1e-5).log()
    def bandpower_outband(self, x):
        # Penalize energy outside 0.5–40 Hz
        X = fft.rfft(x)
        freqs = torch.linspace(0, self.sr/2, X.size(-1), device=x.device)
        mask = (freqs < 0.5) | (freqs > 40.0)
        return (X.abs()**2)[..., mask].mean()
    def forward(self, y_hat, y):
        l1 = self.l1(y_hat, y)
        # Handle 4D input (B, 1, 1, T) -> squeeze to 2D (B, T)
        y_hat_2d = y_hat.squeeze(1).squeeze(1)  # (B, 1, 1, T) -> (B, T)
        y_2d = y.squeeze(1).squeeze(1)  # (B, 1, 1, T) -> (B, T)
        sm = self.stft_mag(y_hat_2d)
        tm = self.stft_mag(y_2d)
        stft = (sm - tm).abs().mean()
        bp = self.bandpower_outband(y_hat_2d)
        return self.w_l1*l1 + self.w_stft*stft + self.w_bp*bp
