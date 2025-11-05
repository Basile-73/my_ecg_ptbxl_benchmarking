# model_v2.py - Enhanced Mamba STFT UNet with MECGE-inspired improvements
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mamba_ssm import Mamba
except Exception as e:
    raise ImportError("Install mamba-ssm: pip install mamba-ssm")


class STFTFrontend(nn.Module):
    """STFT frontend with proper windowing and phase features"""
    def __init__(self, n_fft=256, include_phase=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        self.win_length = n_fft
        self.include_phase = include_phase

        # Register hann window as buffer for device compatibility
        hann_window = torch.hann_window(self.win_length)
        self.register_buffer('hann_window', hann_window)

    def forward(self, x):
        """
        Args:
            x: (B, 1, T) waveform
        Returns:
            features: (B, C, F, T_spec) where C=2 (mag+phase) or 3 (mag+cos+sin)
            X_complex: (B, F, T_spec) complex STFT for reconstruction
        """
        # Apply STFT with proper parameters
        X_complex = torch.stft(
            x.squeeze(1),  # (B, T)
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window,
            center=True,
            onesided=True,
            return_complex=True
        )  # (B, F, T_spec)

        # Magnitude compression
        mag = (X_complex.abs() + 1e-5).pow(0.3)

        if self.include_phase:
            # Phase as cos and sin
            phase = X_complex.angle()
            cos_phase = torch.cos(phase)
            sin_phase = torch.sin(phase)
            features = torch.stack([mag, cos_phase, sin_phase], dim=1)  # (B, 3, F, T_spec)
        else:
            # Just magnitude and phase
            phase = X_complex.angle()
            features = torch.stack([mag, phase], dim=1)  # (B, 2, F, T_spec)

        return features, X_complex

    def inverse(self, X_complex, length):
        """
        Args:
            X_complex: (B, F, T_spec) complex STFT
            length: original signal length
        Returns:
            y: (B, 1, T) reconstructed waveform
        """
        y = torch.istft(
            X_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window,
            center=True,
            onesided=True,
            length=length
        )
        return y.unsqueeze(1)  # (B, 1, T)


class ConvBlockV2(nn.Module):
    """Improved conv block with InstanceNorm, PReLU, and residual connections"""
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, k, stride=s, padding=p)
        self.norm1 = nn.InstanceNorm2d(c_out, affine=True)
        self.prelu1 = nn.PReLU(c_out)

        self.conv2 = nn.Conv2d(c_out, c_out, k, padding=1)
        self.norm2 = nn.InstanceNorm2d(c_out, affine=True)
        self.prelu2 = nn.PReLU(c_out)

        # Residual connection if dimensions match
        self.use_residual = (c_in == c_out and s == 1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.prelu1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.prelu2(out)

        if self.use_residual:
            out = out + identity

        return out


class DenseConvBlock(nn.Module):
    """Dense block with dilated convolutions inspired by MECGE"""
    def __init__(self, base_ch, depth=3):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList()

        for i in range(depth):
            in_ch = base_ch * (i + 1)
            dilation = (2 ** i, 1)
            padding = ((2 ** i), 1)

            layer = nn.Sequential(
                nn.Conv2d(in_ch, base_ch, kernel_size=3, dilation=dilation, padding=padding),
                nn.InstanceNorm2d(base_ch, affine=True),
                nn.PReLU(base_ch)
            )
            self.layers.append(layer)

    def forward(self, x):
        """
        Args:
            x: (B, base_ch, F, T)
        Returns:
            output: (B, base_ch, F, T)
        """
        outputs = [x]

        for i, layer in enumerate(self.layers):
            # Concatenate all previous outputs
            layer_input = torch.cat(outputs, dim=1)
            layer_output = layer(layer_input)
            outputs.append(layer_output)

        # Return only the last layer's output
        return outputs[-1]


class EncoderBlockV2(nn.Module):
    """Encoder block with optional dense connections and frequency downsampling"""
    def __init__(self, c_in, c_out, stride=(2, 1), use_dense=False):
        super().__init__()
        self.conv_block = ConvBlockV2(c_in, c_out, s=stride)
        self.use_dense = use_dense

        if use_dense:
            self.dense_block = DenseConvBlock(c_out, depth=2)

    def forward(self, x):
        x = self.conv_block(x)
        if self.use_dense:
            x = self.dense_block(x)
        return x


class UpBlockV2(nn.Module):
    """Improved upsampling block with frequency-axis upsampling"""
    def __init__(self, c_in, c_skip, c_out, stride=(2, 1)):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in, kernel_size=stride, stride=stride)
        self.conv = ConvBlockV2(c_in + c_skip, c_out)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle shape mismatches - crop if x is larger, pad if x is smaller
        # Time dimension (last dimension)
        if x.size(-1) > skip.size(-1):
            x = x[..., :skip.size(-1)]
        elif x.size(-1) < skip.size(-1):
            x = F.pad(x, (0, skip.size(-1) - x.size(-1)))

        # Frequency dimension (second-to-last dimension)
        if x.size(-2) > skip.size(-2):
            x = x[..., :skip.size(-2), :]
        elif x.size(-2) < skip.size(-2):
            x = F.pad(x, (0, 0, 0, skip.size(-2) - x.size(-2)))

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ComplexMaskHead(nn.Module):
    """Output head for complex mask (real and imaginary components)

    The mask magnitude is stabilized using tanh to bound it within [-1, 1] per component,
    which results in a maximum mask magnitude of sqrt(2). This prevents extreme amplification
    or suppression of spectral components while maintaining the relative phase relationship.
    """
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 2, kernel_size=1)

    def forward(self, x):
        mask = self.conv(x)
        # Apply tanh to stabilize mask magnitude
        mask = torch.tanh(mask)
        return mask


class MambaBlock(nn.Module):
    """Mamba block for sequence modeling (reused from original model)"""
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


class TinyMambaSTFTUNetV2(nn.Module):
    """Enhanced Mamba STFT UNet with MECGE-inspired improvements"""
    def __init__(self, base_ch=32, mamba_dim=64, mamba_depth=3, n_fft=256, include_phase=True, use_dense=True):
        super().__init__()

        # STFT frontend
        self.stft = STFTFrontend(n_fft=n_fft, include_phase=include_phase)

        # Determine input channels based on phase inclusion
        input_channels = 3 if include_phase else 2

        # Encoder path
        self.enc1 = ConvBlockV2(input_channels, base_ch)
        self.enc2 = EncoderBlockV2(base_ch, base_ch * 2, stride=(2, 1), use_dense=use_dense)
        self.enc3 = EncoderBlockV2(base_ch * 2, base_ch * 4, stride=(2, 1), use_dense=use_dense)

        # Bottleneck with Mamba blocks
        self.bridge = nn.Conv2d(base_ch * 4, mamba_dim, 1)
        self.mambas = nn.ModuleList([MambaBlock(mamba_dim) for _ in range(mamba_depth)])
        self.unbridge = nn.Conv2d(mamba_dim, base_ch * 4, 1)

        # Decoder path
        self.up2 = UpBlockV2(base_ch * 4, base_ch * 2, base_ch * 2, stride=(2, 1))
        self.up1 = UpBlockV2(base_ch * 2, base_ch, base_ch, stride=(2, 1))

        # Complex mask head
        self.head = ComplexMaskHead(base_ch)

    def forward(self, x):
        """
        Args:
            x: (B, 1, 1, T) input signal
        Returns:
            y: (B, 1, 1, T) denoised signal
        """
        # Handle 4D input (B, 1, 1, T) - convert to 3D for STFT processing
        B, _, _, T = x.shape
        x = x.squeeze(2)  # (B, 1, 1, T) -> (B, 1, T)

        # Apply STFT frontend
        features, X_complex = self.stft(x)  # features: (B, C, F, T_spec), X_complex: (B, F, T_spec)

        # Encoder path
        s1 = self.enc1(features)
        s2 = self.enc2(s1)
        h = self.enc3(s2)

        # Bottleneck with Mamba
        h = self.bridge(h)
        for blk in self.mambas:
            h = blk(h)
        h = self.unbridge(h)

        # Decoder path
        h = self.up2(h, s2)
        h = self.up1(h, s1)

        # Complex mask head
        mask = self.head(h)  # (B, 2, F, T_spec)
        M_re = mask[:, 0:1, :, :]  # (B, 1, F, T_spec)
        M_im = mask[:, 1:2, :, :]  # (B, 1, F, T_spec)

        # Apply complex multiplication
        M_complex = M_re.squeeze(1) + 1j * M_im.squeeze(1)  # (B, F, T_spec)
        Y_complex = M_complex * X_complex

        # Inverse STFT
        y = self.stft.inverse(Y_complex, length=T)  # (B, 1, T)
        y = y.unsqueeze(2)  # (B, 1, T) -> (B, 1, 1, T)

        return y
