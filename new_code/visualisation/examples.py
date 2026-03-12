"""Cherry-pick denoising examples where the lead model shines.

Usage (from new_code/):
    python -m visualisation.examples --config visualisation/example_config/eu_cherry_pick.yaml
    python -m visualisation.examples --config visualisation/example_config/eu_cherry_pick.yaml --plot 0
    python -m visualisation.examples --config visualisation/example_config/eu_cherry_pick.yaml --plot 0 1 2
"""

import argparse
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecg_noise_factory.noise import NoiseFactory
from utils.getters import get_data_set, get_model
from visualisation.maps import COLOR_MAP, NAME_MAP, plot_font_sizes

# Register CMU Serif font
_FONT_DIR = Path(__file__).resolve().parent.parent.parent / "fonts" / "cm-unicode-0.7.0"
for _ttf in _FONT_DIR.glob("*.ttf"):
    fm.fontManager.addfont(str(_ttf))
plt.rcParams["font.family"] = "CMU Serif"

# Re-use the same sampleset-name helpers used by the Evaluator
from utils.getters import (
    get_sampleset_name,
    get_sampleset_name_european_st_t,
    get_sampleset_name_mitbh_arr,
    get_sampleset_name_mitbh_sin,
    get_sampleset_name_ptbxl,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_temp_config_path(cfg: dict) -> Path:
    """Write a temporary merged config YAML that ``get_data_set`` can read."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="example_cfg_"
    )
    # get_data_set / read_config expect a full config with 'model' + 'training'
    # keys. We inject minimal placeholders so the loader doesn't crash.
    full = dict(cfg)
    if "model" not in full:
        full["model"] = {"type": "unet", "name": "_dummy", "is_stage_2": False, "is_mamba": False}
    if "training" not in full:
        full["training"] = {
            "epochs": 1, "batch_size": cfg.get("batch_size", 32),
            "learning_rate": 1e-3, "loss_function": "MSE",
            "optimizer": "Adam", "scheduler": "ReduceLROnPlateau",
            "early_stopping_patience": 10,
        }
    yaml.dump(full, tmp)
    tmp.flush()
    return Path(tmp.name)


def _get_sampleset_name(cfg: dict) -> str:
    dataset_type = cfg["dataset"]
    dv = cfg["data_volume"]
    sim = cfg["simulation_params"]
    if dataset_type == "synthetic":
        return get_sampleset_name(sim, dv["n_samples_train"], "train")
    elif dataset_type == "mitbih_arrhythmia":
        return get_sampleset_name_mitbh_arr(
            sim["duration"], dv["n_samples_train"], "train"
        )
    elif dataset_type == "mitbih_sinus":
        return get_sampleset_name_mitbh_sin(
            sim["duration"], dv["n_samples_train"], "train"
        )
    elif dataset_type == "european_st_t":
        eu = cfg.get("european_st_t_params", {})
        return get_sampleset_name_european_st_t(
            sim["duration"], dv["n_samples_train"], "train",
            lowcut=eu.get("lowcut", 1.0),
            highcut=eu.get("highcut", 15.0),
            alpha=eu.get("alpha", 2.0),
            ma_window=eu.get("ma_window"),
        )
    elif dataset_type == "ptb_xl":
        ptb = cfg["ptb_xl_params"]
        folds = list(range(1, 9))[: dv["n_folds_train"]]
        return get_sampleset_name_ptbxl(
            split_length=cfg["split_length"],
            folds=folds,
            original_fs=ptb["original_sampling_rate"],
            mode="train",
            lead_index=ptb.get("lead_index", 0),
            select_best_lead=ptb.get("select_best_lead", False),
            remove_bad_labels=ptb.get("remove_bad_labels", False),
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def _load_model(model_entry: dict, split_length: int, device: torch.device):
    """Instantiate a model and load its weights.

    Returns ``(model, is_stage_2, stage1_model | None)``.
    """
    mc = model_entry["model_config"]
    model_type = mc["type"]
    model = get_model(model_type, sequence_length=split_length, model_config=mc)
    state = torch.load(model_entry["weights_path"], map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    stage1_model = None
    if mc.get("is_stage_2", False):
        stage1_model = get_model(
            mc["stage_1_type"], sequence_length=split_length, model_config=mc
        )
        stage1_state = torch.load(mc["stage_1_weights_path"], map_location=device)
        stage1_model.load_state_dict(stage1_state)
        stage1_model.to(device).eval()

    return model, mc.get("is_stage_2", False), stage1_model


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_per_sample_metrics(
    model, eval_loader, device, is_stage_2=False, stage1_model=None
):
    """Return arrays of per-sample RMSE and SNR."""
    rmses, snrs = [], []
    model.eval()
    for noisy, clean in eval_loader:
        noisy = noisy.to(device)
        if is_stage_2:
            pred_1 = stage1_model(noisy)
            input_stage_2 = torch.cat((noisy, pred_1), dim=1)
            denoised = model(input_stage_2)
        else:
            denoised = model(noisy)

        denoised = denoised.detach().cpu().numpy().reshape(len(clean), -1)
        clean = clean.numpy().reshape(len(clean), -1)

        err = clean - denoised
        rmses.extend(np.sqrt((err ** 2).mean(axis=1)))
        snrs.extend(
            10 * np.log10((clean ** 2).mean(axis=1) / (err ** 2).mean(axis=1))
        )
    return np.array(rmses), np.array(snrs)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_samples(per_model_snrs: dict, lead_model: str):
    """Return sample indices sorted by *minimum* SNR advantage of the lead
    model over every other model (descending).

    ``per_model_snrs`` maps model name -> 1-D numpy array of per-sample SNR.
    """
    lead_snr = per_model_snrs[lead_model]
    other_names = [n for n in per_model_snrs if n != lead_model]
    # For each sample, compute the minimum advantage over all other models
    min_advantage = np.full_like(lead_snr, np.inf)
    for name in other_names:
        advantage = lead_snr - per_model_snrs[name]
        min_advantage = np.minimum(min_advantage, advantage)
    # Sort descending (best advantage first)
    return np.argsort(-min_advantage), min_advantage


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict_single(model, noisy_tensor, device, is_stage_2, stage1_model):
    noisy = noisy_tensor.unsqueeze(0).to(device)
    if is_stage_2:
        pred_1 = stage1_model(noisy)
        inp = torch.cat((noisy, pred_1), dim=1)
        return model(inp).squeeze(0).cpu().numpy().reshape(-1)
    return model(noisy).squeeze(0).cpu().numpy().reshape(-1)


def plot_example(
    idx,
    ranked_idx,
    min_advantage,
    eval_dataset,
    models_info,
    per_model_snrs,
    per_model_rmses,
    cfg,
    device,
    output_dir=None,
):
    """Plot a single sample (by rank index) with all model predictions."""
    sample_idx = ranked_idx[idx]
    noisy, clean = eval_dataset[sample_idx]

    clean_np = clean.reshape(-1).numpy()
    noisy_np = noisy.reshape(-1).numpy()

    fs = cfg["simulation_params"]["sampling_rate"]
    t = np.arange(len(clean_np)) / fs

    n_models = len(models_info)
    n_rows = n_models + 1
    side = max(10, n_rows * 3)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(side, side),
        sharex=True,
    )
    if n_rows == 1:
        axes = [axes]

    # --- First subplot: noisy + clean ---
    ax = axes[0]
    ax.plot(t, noisy_np, color="#808080", label="Noisy input", linewidth=1.2)
    ax.plot(t, clean_np, color="green", label="Clean", linewidth=1.6)
    ax.axhline(0, linestyle=":", color="lightgreen", linewidth=1.0)
    ax.set_ylabel("Amplitude", fontsize=plot_font_sizes["axis_labels"])
    ax.legend(fontsize=plot_font_sizes["legend"], loc="upper right")
    ax.set_title(
        f"Sample {sample_idx}  (rank {idx},  min SNR advantage = {min_advantage[sample_idx]:.2f} dB)",
        fontsize=plot_font_sizes["title"],
    )
    ax.tick_params(labelsize=plot_font_sizes["ticks"])

    # --- One subplot per model ---
    for i, entry in enumerate(models_info):
        ax = axes[i + 1]
        name = entry["name"]
        model, is_stage_2, stage1_model = entry["model"], entry["is_stage_2"], entry["stage1_model"]

        denoised = _predict_single(model, noisy, device, is_stage_2, stage1_model)
        color = COLOR_MAP.get(name, "#333333")
        display_name = NAME_MAP.get(name, name)

        rmse_val = per_model_rmses[name][sample_idx]
        snr_val = per_model_snrs[name][sample_idx]

        ax.plot(t, clean_np, color="green", linewidth=1.2, alpha=0.5, label="Clean")
        ax.plot(
            t, denoised, color=color, linewidth=1.6,
            label=f"{display_name}  (RMSE={rmse_val:.4f}, SNR={snr_val:.1f} dB)",
        )
        ax.axhline(0, linestyle=":", color="lightgreen", linewidth=1.0)
        ax.set_ylabel("Amplitude", fontsize=plot_font_sizes["axis_labels"])
        ax.legend(fontsize=plot_font_sizes["legend"], loc="upper right")
        ax.tick_params(labelsize=plot_font_sizes["ticks"])

    axes[-1].set_xlabel("Time (s)", fontsize=plot_font_sizes["axis_labels"])
    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"sample_{sample_idx}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_lead_example(
    idx,
    ranked_idx,
    min_advantage,
    eval_dataset,
    lead_info,
    per_model_snrs,
    per_model_rmses,
    cfg,
    device,
    output_dir=None,
):
    """Single-panel plot: noisy + clean + lead model prediction.

    Uses noise_study-style proportions (8x6) and font sizes.
    """
    sample_idx = ranked_idx[idx]
    noisy, clean = eval_dataset[sample_idx]

    clean_np = clean.reshape(-1).numpy()
    noisy_np = noisy.reshape(-1).numpy()

    fs = cfg["simulation_params"]["sampling_rate"]
    t = np.arange(len(clean_np)) / fs

    lead_name = lead_info["name"]
    model = lead_info["model"]
    is_stage_2 = lead_info["is_stage_2"]
    stage1_model = lead_info["stage1_model"]

    denoised = _predict_single(model, noisy, device, is_stage_2, stage1_model)
    color = COLOR_MAP.get(lead_name, "#333333")
    display_name = NAME_MAP.get(lead_name, lead_name)
    rmse_val = per_model_rmses[lead_name][sample_idx]
    snr_val = per_model_snrs[lead_name][sample_idx]

    plt.rcParams.update({
        'font.size': 36,
        'axes.titlesize': 42,
        'axes.labelsize': 39,
        'xtick.labelsize': 33,
        'ytick.labelsize': 33,
        'legend.fontsize': 36,
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t, noisy_np, color="#808080", label="Noisy input", linewidth=2)
    ax.plot(t, clean_np, color="green", label="Clean", linewidth=2)
    ax.plot(
        t, denoised, color=color, linewidth=2,
        label=f"{display_name}  (RMSE={rmse_val:.4f}, SNR={snr_val:.1f} dB)",
    )
    ax.axhline(0, linestyle=":", color="lightgreen", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"sample_{sample_idx}_lead.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
        print(f"Saved {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cherry-pick denoising examples where the lead model shines."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to a cherry-pick YAML config (e.g. visualisation/example_config/eu_cherry_pick.yaml)",
    )
    parser.add_argument(
        "--plot", type=int, nargs="*", default=None,
        help="Rank indices to plot (0 = sample where lead model has largest advantage). "
             "If omitted, prints the ranking table instead.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_length = cfg["split_length"]
    batch_size = cfg.get("batch_size", 32)

    # --- Build eval dataset (shared across models) ---
    tmp_config_path = _build_temp_config_path(cfg)
    sampleset_name = _get_sampleset_name(cfg)
    scaler_stats = np.loadtxt(f"data/{sampleset_name}_scaler_stats")

    eval_noise_factory = NoiseFactory(
        cfg["noise_paths"]["data_path"],
        cfg["simulation_params"]["sampling_rate"],
        cfg["noise_paths"]["config_path"],
        mode="eval",
        seed=42,
    )
    eval_dataset = get_data_set(
        config_path=tmp_config_path,
        mode="eval",
        noise_factory=eval_noise_factory,
        median=scaler_stats[0],
        iqr=scaler_stats[1],
    )
    eval_loader = DataLoader(eval_dataset, batch_size)

    # --- Load models and compute per-sample metrics ---
    model_entries = cfg["models"]
    models_info = []
    per_model_snrs = {}
    per_model_rmses = {}

    for entry in model_entries:
        name = entry["name"]
        print(f"Loading {name} …")
        model, is_stage_2, stage1_model = _load_model(entry, split_length, device)
        models_info.append({
            "name": name,
            "model": model,
            "is_stage_2": is_stage_2,
            "stage1_model": stage1_model,
        })
        print(f"  Computing metrics for {name} …")
        rmses, snrs = compute_per_sample_metrics(
            model, eval_loader, device, is_stage_2, stage1_model
        )
        per_model_rmses[name] = rmses
        per_model_snrs[name] = snrs
        print(f"  {name}: mean RMSE={rmses.mean():.4f}, mean SNR={snrs.mean():.2f} dB")

    # --- Rank samples ---
    lead_model = cfg["lead_model"]
    assert lead_model in per_model_snrs, (
        f"lead_model '{lead_model}' not found in model names: {list(per_model_snrs)}"
    )
    ranked_idx, min_advantage = rank_samples(per_model_snrs, lead_model)

    # --- Print or plot ---
    if args.plot is None:
        # Print ranking table
        lead_display = NAME_MAP.get(lead_model, lead_model)
        print(f"\nSamples ranked by min SNR advantage of '{lead_display}' over all others:\n")
        header = f"{'Rank':>5}  {'Sample':>6}  {'Advantage (dB)':>14}"
        for name in per_model_snrs:
            display = NAME_MAP.get(name, name)
            header += f"  {display + ' SNR':>18}"
        print(header)
        print("-" * len(header))

        n_show = min(30, len(ranked_idx))
        for rank in range(n_show):
            si = ranked_idx[rank]
            row = f"{rank:>5}  {si:>6}  {min_advantage[si]:>14.2f}"
            for name in per_model_snrs:
                row += f"  {per_model_snrs[name][si]:>18.2f}"
            print(row)
        print(f"\n({len(ranked_idx)} samples total)")
    else:
        # Output folder: visualisation/example_config/<config_stem>/
        config_stem = Path(args.config).stem
        output_dir = Path(args.config).parent / config_stem
        lead_info = next(m for m in models_info if m["name"] == lead_model)
        for rank_idx in args.plot:
            plot_example(
                rank_idx, ranked_idx, min_advantage,
                eval_dataset, models_info,
                per_model_snrs, per_model_rmses,
                cfg, device,
                output_dir=output_dir,
            )
            plot_lead_example(
                rank_idx, ranked_idx, min_advantage,
                eval_dataset, lead_info,
                per_model_snrs, per_model_rmses,
                cfg, device,
                output_dir=output_dir,
            )


if __name__ == "__main__":
    main()
