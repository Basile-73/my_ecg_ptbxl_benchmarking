# loss_v2.py - Enhanced multi-component loss for TinyMambaSTFTUNetV2
import torch
import torch.nn as nn
import torch.fft as fft


def compute_stft_mag(waveform, n_fft, hop_length, win_length, device):
    """
    Compute compressed magnitude STFT.

    Args:
        waveform: (B, T) waveform tensor
        n_fft: FFT size
        hop_length: hop length
        win_length: window length
        device: torch device

    Returns:
        Compressed magnitude tensor
    """
    hann_window = torch.hann_window(win_length).to(device)
    X = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window,
        center=True,
        onesided=True,
        return_complex=True
    )
    # Compressed magnitude
    mag = (X.abs() + 1e-5).pow(0.3)
    return mag


def compute_phase_loss(pred_complex, target_complex):
    """
    Compute phase-aware cosine loss.

    Args:
        pred_complex: (B, F, T) complex STFT of prediction
        target_complex: (B, F, T) complex STFT of target

    Returns:
        Scalar phase loss value
    """
    pred_phase = torch.angle(pred_complex)
    target_phase = torch.angle(target_complex)
    delta_phase = pred_phase - target_phase
    # Cosine similarity loss
    loss = (1 - torch.cos(delta_phase)).mean()
    return loss


def compute_complex_loss(pred_complex, target_complex):
    """
    Compute L1 loss in complex domain.

    Args:
        pred_complex: (B, F, T) complex STFT of prediction
        target_complex: (B, F, T) complex STFT of target

    Returns:
        Scalar complex loss value
    """
    loss = torch.mean(torch.abs(pred_complex - target_complex))
    return loss


def si_sdr(pred, target, eps=1e-8):
    """
    Compute scale-invariant signal-to-distortion ratio (SI-SDR).

    Args:
        pred: (B, T) predicted waveform
        target: (B, T) target waveform
        eps: small epsilon for numerical stability

    Returns:
        Scalar SI-SDR value (in dB, higher is better)
    """
    # Zero-mean normalization
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # Compute scaling factor
    alpha = (pred * target).sum(dim=-1, keepdim=True) / (target ** 2).sum(dim=-1, keepdim=True).clamp(min=eps)

    # Scaled target (signal)
    target_scaled = alpha * target

    # Distortion
    distortion = pred - target_scaled

    # SI-SDR computation
    signal_power = (target_scaled ** 2).sum(dim=-1)
    distortion_power = (distortion ** 2).sum(dim=-1).clamp(min=eps)

    si_sdr_value = 10 * torch.log10(signal_power / distortion_power + eps)

    # Average across batch
    return si_sdr_value.mean()


class EnhancedSTFTLoss(nn.Module):
    """
    Enhanced multi-component loss combining time-domain, multi-resolution STFT,
    phase, complex, consistency, and bandpower losses.
    """
    def __init__(
        self,
        sr=250,
        w_time=1.0,
        w_mr_stft=1.0,
        w_phase=1.0,
        w_complex=1.0,
        w_consistency=1.0,
        w_bandpower=0.2,
        w_sisdr=0.0,
        use_sisdr=False,
        mr_stft_configs=None
    ):
        super().__init__()
        self.sr = sr
        self.w_time = w_time
        self.w_mr_stft = w_mr_stft
        self.w_phase = w_phase
        self.w_complex = w_complex
        self.w_consistency = w_consistency
        self.w_bandpower = w_bandpower
        self.w_sisdr = w_sisdr
        self.use_sisdr = use_sisdr

        # Default multi-resolution STFT configurations
        if mr_stft_configs is None:
            self.mr_stft_configs = [(128, 32), (256, 64), (512, 128), (1024, 256)]
        else:
            self.mr_stft_configs = mr_stft_configs

        # Time-domain loss
        self.l1 = nn.L1Loss()

        # Window cache for STFT computations
        self.window_cache = {}

    def get_window(self, win_length, device):
        """
        Get or create cached Hann window for STFT computations.

        Args:
            win_length: window length
            device: torch device

        Returns:
            Cached Hann window tensor
        """
        key = (win_length, device)
        if key not in self.window_cache:
            self.window_cache[key] = torch.hann_window(win_length).to(device)
        return self.window_cache[key]

    def multi_resolution_stft_loss(self, y_pred, y_true):
        """
        Compute multi-resolution STFT loss across multiple scales.

        Args:
            y_pred: (B, T) predicted waveform
            y_true: (B, T) target waveform

        Returns:
            Average loss across all resolutions
        """
        total_loss = 0
        device = y_pred.device

        for n_fft, hop_length in self.mr_stft_configs:
            win_length = n_fft

            # Get cached window
            hann_window = self.get_window(win_length, device)

            # Compute compressed magnitudes
            mag_pred = self._compute_stft_mag_cached(y_pred, n_fft, hop_length, win_length, hann_window)
            mag_true = self._compute_stft_mag_cached(y_true, n_fft, hop_length, win_length, hann_window)

            # L1 loss between compressed magnitudes
            loss = torch.mean(torch.abs(mag_pred - mag_true))
            total_loss += loss

        # Average across resolutions
        return total_loss / len(self.mr_stft_configs)

    def _compute_stft_mag_cached(self, waveform, n_fft, hop_length, win_length, window):
        """
        Compute compressed magnitude STFT using cached window.

        Args:
            waveform: (B, T) waveform tensor
            n_fft: FFT size
            hop_length: hop length
            win_length: window length
            window: precomputed window tensor

        Returns:
            Compressed magnitude tensor
        """
        X = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            onesided=True,
            return_complex=True
        )
        # Compressed magnitude
        mag = (X.abs() + 1e-5).pow(0.3)
        return mag

    def stft_consistency_loss(self, y_pred, n_fft=256, hop_length=64):
        """
        Compute STFT/iSTFT round-trip consistency loss.

        Args:
            y_pred: (B, T) predicted waveform
            n_fft: FFT size
            hop_length: hop length

        Returns:
            Scalar consistency loss value
        """
        win_length = n_fft
        device = y_pred.device
        hann_window = self.get_window(win_length, device)

        # Forward STFT
        Y_pred_complex = torch.stft(
            y_pred,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=hann_window,
            center=True,
            onesided=True,
            return_complex=True
        )

        # Inverse STFT
        y_recon = torch.istft(
            Y_pred_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=hann_window,
            center=True,
            onesided=True,
            length=y_pred.size(-1)
        )

        # Forward STFT again
        Y_recon_complex = torch.stft(
            y_recon,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=hann_window,
            center=True,
            onesided=True,
            return_complex=True
        )

        # L1 distance in complex domain
        loss = torch.mean(torch.abs(Y_pred_complex - Y_recon_complex))
        return loss

    def bandpower_outband(self, x):
        """
        Penalize energy outside physiological ECG band (0.5-40 Hz).

        Args:
            x: (B, T) waveform

        Returns:
            Scalar bandpower penalty value
        """
        X = fft.rfft(x)
        freqs = torch.linspace(0, self.sr / 2, X.size(-1), device=x.device)
        mask = (freqs < 0.5) | (freqs > 40.0)
        power = (X.abs() ** 2)[..., mask].mean()
        return power

    def forward(self, y_pred, y_true):
        """
        Compute combined loss.

        Args:
            y_pred: (B, 1, 1, T) predicted signal
            y_true: (B, 1, 1, T) target signal

        Returns:
            Total weighted loss
        """
        # Handle 4D input by squeezing to 2D
        y_pred_2d = y_pred.squeeze(1).squeeze(1)  # (B, T)
        y_true_2d = y_true.squeeze(1).squeeze(1)  # (B, T)

        # Loss Component 1: Time-Domain Loss
        loss_time = self.l1(y_pred, y_true)

        # Loss Component 2: Multi-Resolution STFT Loss
        loss_mr_stft = self.multi_resolution_stft_loss(y_pred_2d, y_true_2d)

        # Compute STFT for phase and complex losses (reuse computation)
        n_fft = 256
        hop_length = 64
        win_length = n_fft
        device = y_pred_2d.device
        hann_window = self.get_window(win_length, device)

        pred_complex = torch.stft(
            y_pred_2d,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=hann_window,
            center=True,
            onesided=True,
            return_complex=True
        )

        true_complex = torch.stft(
            y_true_2d,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=hann_window,
            center=True,
            onesided=True,
            return_complex=True
        )

        # Loss Component 3: Phase Loss
        loss_phase = compute_phase_loss(pred_complex, true_complex)

        # Loss Component 4: Complex Loss
        loss_complex = compute_complex_loss(pred_complex, true_complex)

        # Loss Component 5: Consistency Loss
        loss_consistency = self.stft_consistency_loss(y_pred_2d)

        # Loss Component 6: Bandpower Penalty
        loss_bandpower = self.bandpower_outband(y_pred_2d)

        # Combine all losses with weights
        total_loss = (
            self.w_time * loss_time +
            self.w_mr_stft * loss_mr_stft +
            self.w_phase * loss_phase +
            self.w_complex * loss_complex +
            self.w_consistency * loss_consistency +
            self.w_bandpower * loss_bandpower
        )

        # Optional Loss Component 7: SI-SDR Loss
        if self.use_sisdr or self.w_sisdr > 0:
            loss_sisdr = -si_sdr(y_pred_2d, y_true_2d)
            total_loss = total_loss + self.w_sisdr * loss_sisdr

        return total_loss
