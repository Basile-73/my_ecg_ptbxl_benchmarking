"""Plot example signals from the EuropeanSTTDataset for alpha=None vs alpha=2.

Run from new_code/:
    python -m visualisation.plot_eu_examples
"""

import sys
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import EuropeanSTTDataset
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

CONFIG_PATH = Path("experiments/reproduce_eu_smooth/data_configs/eu.yaml")
OUTPUT_DIR = Path("visualisation/output/eu_examples")
SAMPLE_INDICES = [0, 1, 2]
SPLIT_LENGTH = 3600  # 10 s at 360 Hz


def _apply_alpha(signal: np.ndarray, alpha: float) -> np.ndarray:
    return np.sign(signal) * np.log1p(alpha * np.abs(signal))


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
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    eu = cfg["european_st_t_params"]
    sim = cfg["simulation_params"]
    dv = cfg["data_volume"]
    noise_paths = cfg["noise_paths"]

    fs = sim["sampling_rate"]
    duration = eu["duration"]
    assert SPLIT_LENGTH <= duration * fs and (duration * fs) % SPLIT_LENGTH == 0

    noise_factory = NoiseFactory(
        data_path=noise_paths["data_path"],
        sampling_rate=fs,
        config_path=noise_paths["config_path"],
        mode="train",
        seed=42,
    )

    dataset = EuropeanSTTDataset(
        n_samples=dv["n_samples_train"],
        noise_factory=noise_factory,
        duration=duration,
        split_length=SPLIT_LENGTH,
        data_path=eu["data_path"],
        save_clean_samples=False,
        highcut=eu["highcut"],
        lowcut=eu["lowcut"],
        alpha=eu["alpha"],  # None
        ma_window=eu["ma_window"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t = np.arange(SPLIT_LENGTH) / fs

    for idx in SAMPLE_INDICES:
        base_signal = dataset.samples[idx]
        alpha_signal = _apply_alpha(base_signal, 2.0)

        _plot_single(
            t, base_signal, color="#264653",
            output_path=OUTPUT_DIR / f"sample_{idx}_alpha_none.png",
        )
        _plot_single(
            t, alpha_signal, color="#e76f51",
            output_path=OUTPUT_DIR / f"sample_{idx}_alpha_2.png",
        )


if __name__ == "__main__":
    main()
