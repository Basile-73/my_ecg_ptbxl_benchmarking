"""Plot the European ST-T samples with the highest beat-to-beat irregularity
alongside a synthetic counterpart of the same length.

Run from new_code/:
    python -m visualisation.plot_eu_irregular_vs_synthetic --top-n 5 --pool-size 4096
"""

import argparse
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import EuropeanSTTDataset, SyntheticEcgDataset
from ecg_noise_factory.noise import NoiseFactory

_FONT_DIR = Path(__file__).resolve().parent.parent.parent / "fonts" / "cm-unicode-0.7.0"
for _ttf in _FONT_DIR.glob("*.ttf"):
    fm.fontManager.addfont(str(_ttf))
plt.rcParams["font.family"] = "CMU Serif"


_PLOT_RCPARAMS = {
    "font.size": 36,
    "axes.titlesize": 42,
    "axes.labelsize": 39,
    "xtick.labelsize": 33,
    "ytick.labelsize": 33,
    "legend.fontsize": 36,
}

CONFIG_PATH = Path("visualisation/example_config/eu_strong_multi.yaml")
OUTPUT_DIR = Path("visualisation/output/eu_irregular_vs_synthetic")
SPLIT_LENGTH = 3600  # 10 s at 360 Hz
CLEAN_COLOR = "#ABABAB"
BEAT_WINDOW_S = 0.6

_SYNTHETIC_SIM_PARAMS = {
    "duration": 10,
    "sampling_rate": 360,
    "heart_rate": [60, 80],
    "heart_rate_std": 5,
    "lfhfratio": 0.001,
    "means_ai": [1.2, -5, 30, -7.5, 0.75],
    "stds_ai": [0.6, 0.2, 0.0, 1, 0.35],
    "means_bi": [0.25, 0.1, 0.1, 0.1, 0.4],
    "stds_bi": [0.1, 0.1, 0.0, 0.0, 0.0],
}


def beat_dissimilarity(signal: np.ndarray, fs: int, beat_window_s: float) -> float:
    """Higher = beats inside this record differ more from each other."""
    try:
        _, info = nk.ecg_peaks(signal, sampling_rate=fs)
        peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        return -np.inf
    half = int(beat_window_s * fs / 2)
    beats = [
        signal[p - half : p + half]
        for p in peaks
        if p - half >= 0 and p + half < len(signal)
    ]
    if len(beats) < 3:
        return -np.inf
    beats = np.stack(beats)
    # Amplitude-normalise so the score captures shape variability, not drift.
    norms = np.linalg.norm(beats, axis=1, keepdims=True) + 1e-9
    beats = beats / norms
    template = beats.mean(axis=0)
    return float(np.mean(np.linalg.norm(beats - template, axis=1)))


def _plot_boxplot(eu_scores: np.ndarray, synth_scores: np.ndarray, output_path: Path):
    plt.rcParams.update(_PLOT_RCPARAMS)
    eu_valid = eu_scores[np.isfinite(eu_scores)]
    synth_valid = synth_scores[np.isfinite(synth_scores)]
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(
        [eu_valid, synth_valid],
        labels=["European ST-T", "Synthetic"],
        patch_artist=True,
        showfliers=True,
    )
    for patch, color in zip(bp["boxes"], ["#264653", "#e76f51"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Beat dissimilarity")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(
        f"Saved {output_path} "
        f"(EU n={len(eu_valid)} median={np.median(eu_valid):.4f}, "
        f"synth n={len(synth_valid)} median={np.median(synth_valid):.4f})"
    )


def _plot_single(t, signal, color, output_path):
    plt.rcParams.update(_PLOT_RCPARAMS)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, signal, color=color, linewidth=2)
    ax.axhline(0, linestyle=":", color="lightgreen", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=4096)
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    eu = cfg["european_st_t_params"]
    sim = cfg["simulation_params"]
    noise_paths = cfg["noise_paths"]

    fs = sim["sampling_rate"]
    duration = eu["duration"]
    assert SPLIT_LENGTH <= duration * fs and (duration * fs) % SPLIT_LENGTH == 0

    # LengthExperimentDataset reshapes each loaded record of length duration*fs
    # into (duration*fs / SPLIT_LENGTH) segments, so request proportionally fewer.
    segments_per_record = (duration * fs) // SPLIT_LENGTH
    n_records = max(1, -(-args.pool_size // segments_per_record))  # ceil

    noise_factory = NoiseFactory(
        data_path=noise_paths["data_path"],
        sampling_rate=fs,
        config_path=noise_paths["config_path"],
        mode="train",
        seed=42,
    )

    eu_dataset = EuropeanSTTDataset(
        n_samples=n_records,
        noise_factory=noise_factory,
        duration=duration,
        split_length=SPLIT_LENGTH,
        data_path=eu["data_path"],
        save_clean_samples=False,
        highcut=eu["highcut"],
        lowcut=eu["lowcut"],
        alpha=eu.get("alpha"),
        ma_window=eu.get("ma_window"),
    )

    synthetic_dataset = SyntheticEcgDataset(
        simulation_params=_SYNTHETIC_SIM_PARAMS,
        n_samples=args.pool_size,
        noise_factory=noise_factory,
        save_clean_samples=False,
    )

    eu_scores = np.array([
        beat_dissimilarity(eu_dataset.samples[i], fs, BEAT_WINDOW_S)
        for i in range(eu_dataset.samples.shape[0])
    ])
    synth_scores = np.array([
        beat_dissimilarity(synthetic_dataset.samples[i], fs, BEAT_WINDOW_S)
        for i in range(synthetic_dataset.samples.shape[0])
    ])
    top_idx = np.argsort(-eu_scores)[: args.top_n]

    print("Top irregular ST-T samples:")
    for rank, idx in enumerate(top_idx):
        print(f"  rank {rank}: idx={idx}, score={eu_scores[idx]:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t = np.arange(SPLIT_LENGTH) / fs

    for rank, idx in enumerate(top_idx):
        _plot_single(
            t,
            eu_dataset.samples[idx],
            color=CLEAN_COLOR,
            output_path=OUTPUT_DIR / f"rank_{rank}_eu_idx_{idx}.png",
        )
        _plot_single(
            t,
            synthetic_dataset.samples[rank],
            color=CLEAN_COLOR,
            output_path=OUTPUT_DIR / f"rank_{rank}_synthetic.png",
        )

    _plot_boxplot(
        eu_scores,
        synth_scores,
        output_path=OUTPUT_DIR / "beat_dissimilarity_boxplot.png",
    )


if __name__ == "__main__":
    main()
