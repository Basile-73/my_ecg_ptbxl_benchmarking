"""
GPU memory profiling for MECGE and UNet Mamba Block models.

Measures peak GPU memory as a function of signal length (1-5s at 360 Hz)
for both inference and training, and produces a side-by-side comparison plot.

Usage:
    python memory_profile_plot.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'MECGE'))

import gc
import torch
import torch.nn.functional as F
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from maps import COLOR_MAP, NAME_MAP, plot_font_sizes as _base_font_sizes

FONT_SCALE = 1.4
plot_font_sizes = {k: int(v * FONT_SCALE) for k, v in _base_font_sizes.items()}

SAMPLING_RATE = 360
BATCH_SIZE = 2
# Signal lengths: multiples of 40 from 360 to 1800 (1s to 5s at 360 Hz)
SIGNAL_LENGTHS = list(range(360, 1801, 40))

MODELS_TO_PROFILE = {
    'mecge': {
        'color': COLOR_MAP['mecge'],
        'name': NAME_MAP['mecge'],
    },
    'mamba1_3blocks': {
        'color': COLOR_MAP['mamba1_3blocks'],
        'name': NAME_MAP['mamba1_3blocks'],
    },
}


def build_mecge(device):
    from models.MECGE.MECGE import MECGE
    config_path = Path(__file__).parent.parent / 'models' / 'MECGE' / 'config' / 'MECGE_phase.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model = MECGE(config).to(device)
    return model


def build_unet_mamba(input_length, device):
    from models.UNet.Stage1_UNet_Mamba_Block import UNetMambaBlock
    model = UNetMambaBlock(
        input_length=input_length,
        mamba_type='Mamba1',
        n_blocks=3,
    ).to(device)
    return model


def clear_gpu(device):
    """Aggressively free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def measure_peak_memory(fn, device):
    """Run fn() and return peak GPU memory in MB, or None on OOM."""
    clear_gpu(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    try:
        fn()
        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        return peak_bytes / (1024 ** 2)
    except torch.cuda.OutOfMemoryError:
        clear_gpu(device)
        return None


def warmup_gpu(device):
    """Run a dummy forward+backward to trigger CUDA context and kernel caching."""
    print("Warming up GPU (CUDA context + kernel cache)...")
    m = torch.nn.Linear(64, 64).to(device)
    x = torch.randn(2, 64, device=device)
    loss = m(x).sum()
    loss.backward()
    del m, x, loss
    clear_gpu(device)


def warmup_model(model, dummy_fn, device):
    """Run a model once to trigger Triton/CUDA kernel JIT compilation."""
    dummy_fn()
    model.zero_grad()
    clear_gpu(device)


def profile_mecge(device):
    """Profile MECGE for all signal lengths."""
    inference_mem = []
    training_mem = []

    # Warmup: trigger Mamba SSM Triton kernel compilation
    print("  Warming up MECGE kernels...", flush=True)
    warmup_model_obj = build_mecge(device)
    warmup_model_obj.train()
    wc = torch.randn(BATCH_SIZE, 1, 360, device=device)
    wn = torch.randn(BATCH_SIZE, 1, 360, device=device)
    warmup_model(warmup_model_obj, lambda: warmup_model_obj.get_loss(wc, wn).backward(), device)
    del warmup_model_obj, wc, wn
    clear_gpu(device)

    for length in SIGNAL_LENGTHS:
        print(f"  MECGE length={length} ({length/SAMPLING_RATE:.2f}s)...", end=" ", flush=True)

        # --- Inference ---
        model = build_mecge(device)
        model.eval()
        x_infer = torch.randn(BATCH_SIZE, 1, length, device=device)

        def run_inference():
            with torch.no_grad():
                model._denoise_chunk(x_infer)

        mem = measure_peak_memory(run_inference, device)
        inference_mem.append(mem)
        del x_infer
        print(f"inf={mem:.0f}MB" if mem else "inf=OOM", end=" ", flush=True)

        # --- Training ---
        model.train()
        clean = torch.randn(BATCH_SIZE, 1, length, device=device)
        noisy = torch.randn(BATCH_SIZE, 1, length, device=device)

        def run_training():
            loss = model.get_loss(clean, noisy)
            loss.backward()

        mem = measure_peak_memory(run_training, device)
        training_mem.append(mem)
        print(f"train={mem:.0f}MB" if mem else "train=OOM")

        del clean, noisy, model
        clear_gpu(device)

    return inference_mem, training_mem


def profile_unet_mamba(device):
    """Profile UNet Mamba Block for all signal lengths."""
    inference_mem = []
    training_mem = []

    # Warmup: trigger Mamba kernel compilation
    print("  Warming up UNet-Mamba kernels...", flush=True)
    wm = build_unet_mamba(360, device)
    wm.train()
    wx = torch.randn(BATCH_SIZE, 1, 1, 360, device=device)
    warmup_model(wm, lambda: F.mse_loss(wm(wx), wx).backward(), device)
    del wm, wx
    clear_gpu(device)

    for length in SIGNAL_LENGTHS:
        print(f"  UNet-Mamba length={length} ({length/SAMPLING_RATE:.2f}s)...", end=" ", flush=True)

        model = build_unet_mamba(length, device)
        x = torch.randn(BATCH_SIZE, 1, 1, length, device=device)

        # --- Inference ---
        model.eval()

        def run_inference():
            with torch.no_grad():
                model(x)

        mem = measure_peak_memory(run_inference, device)
        inference_mem.append(mem)
        print(f"inf={mem:.0f}MB" if mem else "inf=OOM", end=" ", flush=True)

        # --- Training ---
        model.train()
        target = torch.randn(BATCH_SIZE, 1, 1, length, device=device)

        def run_training():
            out = model(x)
            loss = F.mse_loss(out, target)
            loss.backward()

        mem = measure_peak_memory(run_training, device)
        training_mem.append(mem)
        print(f"train={mem:.0f}MB" if mem else "train=OOM")

        del model, x, target
        clear_gpu(device)

    return inference_mem, training_mem


def make_plot(results, save_path=None):
    """Create side-by-side inference/training memory plots."""
    seconds = np.array(SIGNAL_LENGTHS) / SAMPLING_RATE

    fig, (ax_inf, ax_train) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for model_key, data in results.items():
        color = MODELS_TO_PROFILE[model_key]['color']
        name = MODELS_TO_PROFILE[model_key]['name']

        # Filter out OOM (None) values
        for ax, key in [(ax_inf, 'inference'), (ax_train, 'training')]:
            valid = [(s, m) for s, m in zip(seconds, data[key]) if m is not None]
            if valid:
                xs, ys = zip(*valid)
                ax.plot(xs, ys, color=color, label=name,
                        marker='o', markersize=3, linewidth=2)

    for ax, title in [(ax_inf, 'Inference'), (ax_train, 'Training')]:
        ax.set_title(title, fontsize=plot_font_sizes['title'])
        ax.set_xlabel('Signal length (s)', fontsize=plot_font_sizes['axis_labels'])
        ax.tick_params(labelsize=plot_font_sizes['ticks'])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=plot_font_sizes['legend'])

    ax_inf.set_ylabel('Peak GPU memory (MB)', fontsize=plot_font_sizes['axis_labels'])

    # Description (for use elsewhere):
    # Peak GPU memory allocation as a function of ECG signal length
    # (1-5 s at 360 Hz, batch size 2). Left: inference (forward pass, no
    # gradients). Right: training (forward + backward pass). Both models scale
    # roughly linearly with signal length, but MECGE training requires
    # substantially more memory than UNet Mamba1-3B due to its STFT-domain
    # processing and dual decoder branches.

    fig.suptitle(f'GPU Memory vs Signal Length (batch size={BATCH_SIZE})',
                 fontsize=plot_font_sizes['title'], y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


if __name__ == '__main__':
    assert torch.cuda.is_available(), "CUDA is required for GPU memory profiling"
    device = torch.device('cuda')

    free_mem = torch.cuda.mem_get_info(device)[0] / (1024**3)
    total_mem = torch.cuda.mem_get_info(device)[1] / (1024**3)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Memory: {free_mem:.1f} GiB free / {total_mem:.1f} GiB total")

    if free_mem < 1.0:
        print(f"\nWARNING: Only {free_mem:.1f} GiB free. Results may be incomplete due to OOM.")

    print(f"\nProfiling with batch_size={BATCH_SIZE}, "
          f"lengths={SIGNAL_LENGTHS[0]}-{SIGNAL_LENGTHS[-1]} samples "
          f"({SIGNAL_LENGTHS[0]/SAMPLING_RATE:.1f}-{SIGNAL_LENGTHS[-1]/SAMPLING_RATE:.1f}s)")

    warmup_gpu(device)

    results = {}

    print("\nProfiling UNet Mamba1-3B...")
    inf_mem, train_mem = profile_unet_mamba(device)
    results['mamba1_3blocks'] = {'inference': inf_mem, 'training': train_mem}

    print("\nProfiling MECGE...")
    inf_mem, train_mem = profile_mecge(device)
    results['mecge'] = {'inference': inf_mem, 'training': train_mem}

    for model_key, data in results.items():
        name = MODELS_TO_PROFILE[model_key]['name']
        valid_inf = [m for m in data['inference'] if m is not None]
        valid_train = [m for m in data['training'] if m is not None]
        print(f"\n{name}:")
        if valid_inf:
            print(f"  Inference:  {min(valid_inf):.1f} - {max(valid_inf):.1f} MB")
        else:
            print(f"  Inference:  all OOM")
        if valid_train:
            print(f"  Training:   {min(valid_train):.1f} - {max(valid_train):.1f} MB")
        else:
            print(f"  Training:   all OOM")

    save_path = Path(__file__).parent / 'memory_profile.png'
    make_plot(results, save_path=save_path)
