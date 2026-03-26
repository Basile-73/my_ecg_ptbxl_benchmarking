"""
Record-level evaluation comparing two experiments across split lengths.

For models trained at shorter split_lengths, full-length records are chunked,
denoised per-chunk, and reassembled before computing metrics on the full record.
This avoids the non-additivity bias of averaging per-segment SNR/RMSE.

Usage:
    python evaluate_record_level.py --exp_1 length_syn --exp_2 length_syn_mamba_no_comp
"""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from train_multiple import get_model_weights_name
from utils.read_configs import get_configs
from utils.getters import (
    get_model,
    get_data_set,
    get_percentiles,
    get_sampleset_name,
    get_sampleset_name_european_st_t,
    get_sampleset_name_mitbh_arr,
    get_sampleset_name_mitbh_sin,
    get_sampleset_name_ptbxl,
)
from ecg_noise_factory.noise import NoiseFactory
from visualisation.maps import COLOR_MAP, NAME_MAP, OUR_MODELS, plot_font_sizes

# Register CMU Serif font
_REPO_ROOT = Path(__file__).resolve().parent.parent
_FONT_DIR = _REPO_ROOT / "fonts" / "cm-unicode-0.7.0"
for _ttf in _FONT_DIR.glob("*.ttf"):
    fm.fontManager.addfont(str(_ttf))
plt.rcParams["font.family"] = "CMU Serif"

FONT_SCALE = 2.0
SCALED_FONT_SIZES = {k: v * FONT_SCALE for k, v in plot_font_sizes.items()}


def _get_experiment_model_name(model_configs):
    """Extract the unique model name from an experiment's configs.

    Asserts all configs in the experiment use the same model name.
    Returns the name, falling back to stripping a trailing '_N' suffix
    for map lookup.
    """
    names = {c["model"]["name"] for c in model_configs}
    assert len(names) == 1, (
        f"Expected one model type per experiment, got: {names}"
    )
    return names.pop()


def _map_lookup(model_name):
    """Look up display name, color, and (ours) flag from maps.

    Tries the raw name first, then strips a trailing '_N' digit suffix.
    """
    # Try exact match, then strip trailing _<digit(s)>
    for candidate in [model_name, model_name.rsplit("_", 1)[0]]:
        if candidate in COLOR_MAP:
            display = NAME_MAP.get(candidate, candidate)
            if candidate in OUR_MODELS:
                display = f"{display} (ours)"
            return display, COLOR_MAP[candidate]
    # Fallback
    return model_name, None


def _resolve_sampleset_name(cfg):
    """Dispatch to the correct sampleset name builder based on dataset type."""
    dataset_type = cfg["dataset"]
    dv = cfg["data_volume"]
    sim = cfg["simulation_params"]
    if dataset_type == "synthetic":
        return get_sampleset_name(sim, dv["n_samples_train"], "train")
    elif dataset_type == "mitbih_arrhythmia":
        return get_sampleset_name_mitbh_arr(sim["duration"], dv["n_samples_train"], "train")
    elif dataset_type == "mitbih_sinus":
        return get_sampleset_name_mitbh_sin(sim["duration"], dv["n_samples_train"], "train")
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


def get_record_length(data_config):
    return int(
        data_config["simulation_params"]["duration"]
        * data_config["simulation_params"]["sampling_rate"]
    )


def load_eval_records(data_config):
    """Load eval dataset at full record length. Returns list of (noisy, clean) tensors."""
    record_length = get_record_length(data_config)
    sampling_rate = data_config["simulation_params"]["sampling_rate"]

    temp_config = dict(data_config)
    temp_config["split_length"] = record_length

    fd, temp_path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.dump(temp_config, f)

    try:
        train_sampleset = _resolve_sampleset_name(data_config)
        scaler_stats = np.loadtxt(f"data/{train_sampleset}_scaler_stats")

        eval_noise_factory = NoiseFactory(
            data_config["noise_paths"]["data_path"],
            sampling_rate,
            data_config["noise_paths"]["config_path"],
            mode="eval",
            seed=42,
        )

        eval_dataset = get_data_set(
            config_path=temp_path,
            mode="eval",
            noise_factory=eval_noise_factory,
            median=scaler_stats[0],
            iqr=scaler_stats[1],
        )

        records = []
        for i in range(len(eval_dataset)):
            noisy, clean = eval_dataset[i]
            records.append((noisy, clean))
    finally:
        os.unlink(temp_path)

    return records


def load_model(merged_config, experiment_name, device):
    """Load a trained model from its merged (model + data) config."""
    split_length = merged_config["split_length"]
    model_type = merged_config["model"]["type"]
    model_config = merged_config["model"]

    model = get_model(model_type, sequence_length=split_length, model_config=model_config)

    weights_name = get_model_weights_name(merged_config, experiment_name)
    state = torch.load(f"model_weights/{weights_name}.pth", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def denoise_record(model, noisy, split_length, device):
    """
    Denoise a full-length record by chunking into non-overlapping segments.

    Args:
        noisy: [1, 1, record_length] tensor from dataset
        split_length: model's training split length
    Returns:
        [1, 1, record_length] denoised tensor
    """
    record_length = noisy.shape[-1]
    assert record_length % split_length == 0, (
        f"Record length {record_length} not divisible by split_length {split_length}"
    )
    n_chunks = record_length // split_length

    # Batch all chunks: [n_chunks, 1, 1, split_length]
    chunks = noisy.reshape(-1).reshape(n_chunks, 1, 1, split_length).to(device)

    with torch.no_grad():
        denoised_chunks = model(chunks)

    return denoised_chunks.cpu().reshape(1, 1, record_length)


def compute_record_metrics(clean_np, denoised_np):
    """Compute SNR (dB), RMSE, and PCC on a single full-length record."""
    err = clean_np - denoised_np
    mse = (err**2).mean()
    rmse = np.sqrt(mse)
    snr = 10 * np.log10((clean_np**2).mean() / mse)
    pcc = np.corrcoef(clean_np, denoised_np)[0, 1]
    return rmse, snr, pcc


def evaluate_experiment(experiment_name, model_configs, data_config, records, device):
    """Evaluate all models from one experiment on preloaded records.

    Returns:
        dict mapping split_length -> {snr_mean, snr_ci, rmse_mean, rmse_ci, pcc_mean, pcc_ci}
    """
    results = {}

    for model_config in model_configs:
        merged = {**model_config, **data_config}
        split_length = merged["split_length"]
        model_name = get_model_weights_name(merged, experiment_name)
        print(f"  {model_name} (split_length={split_length})")

        model = load_model(merged, experiment_name, device)

        rmses, snrs, pccs = [], [], []
        for noisy, clean in records:
            denoised = denoise_record(model, noisy, split_length, device)
            clean_np = clean.numpy().reshape(-1)
            denoised_np = denoised.numpy().reshape(-1)
            rmse, snr, pcc = compute_record_metrics(clean_np, denoised_np)
            rmses.append(rmse)
            snrs.append(snr)
            pccs.append(pcc)

        results[split_length] = {
            "snr_mean": np.mean(snrs),
            "snr_ci": get_percentiles(snrs),
            "rmse_mean": np.mean(rmses),
            "rmse_ci": get_percentiles(rmses),
            "pcc_mean": np.mean(pccs),
            "pcc_ci": get_percentiles(pccs),
        }
        r = results[split_length]
        print(
            f"    SNR={r['snr_mean']:.2f} dB, "
            f"RMSE={r['rmse_mean']:.4f}, "
            f"PCC={r['pcc_mean']:.4f}"
        )

    return results


def plot_comparison(results_1, results_2, model_name_1, model_name_2,
                    sampling_rate, output_path):
    split_lengths = sorted(set(results_1.keys()) & set(results_2.keys()))
    x_seconds = [sl / sampling_rate for sl in split_lengths]

    label_1, color_1 = _map_lookup(model_name_1)
    label_2, color_2 = _map_lookup(model_name_2)
    color_1 = color_1 or "tab:blue"
    color_2 = color_2 or "tab:orange"

    fig, ax = plt.subplots(figsize=(10, 6))

    means_1 = [results_1[sl]["snr_mean"] for sl in split_lengths]
    ci_lo_1 = [results_1[sl]["snr_ci"][0] for sl in split_lengths]
    ci_hi_1 = [results_1[sl]["snr_ci"][1] for sl in split_lengths]

    means_2 = [results_2[sl]["snr_mean"] for sl in split_lengths]
    ci_lo_2 = [results_2[sl]["snr_ci"][0] for sl in split_lengths]
    ci_hi_2 = [results_2[sl]["snr_ci"][1] for sl in split_lengths]

    # Fill the space between the two curves in green
    ax.fill_between(x_seconds, means_1, means_2, alpha=0.15, color="green")

    ax.plot(x_seconds, means_1, "o-", color=color_1, label=label_1)
    ax.fill_between(x_seconds, ci_lo_1, ci_hi_1, alpha=0.2, color=color_1)

    ax.plot(x_seconds, means_2, "o-", color=color_2, label=label_2)
    ax.fill_between(x_seconds, ci_lo_2, ci_hi_2, alpha=0.2, color=color_2)

    ax.set_xlabel("Split Length (seconds)", fontsize=SCALED_FONT_SIZES["axis_labels"],
                  fontweight="bold")
    ax.set_ylabel("SNR (dB)", fontsize=SCALED_FONT_SIZES["axis_labels"],
                  fontweight="bold")
    ax.set_title("Record-Level SNR", fontsize=SCALED_FONT_SIZES["title"],
                 fontweight="bold")
    ax.legend(fontsize=SCALED_FONT_SIZES["legend"])
    ax.tick_params(axis="both", labelsize=SCALED_FONT_SIZES["ticks"])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def save_results_csv(results_1, results_2, label_1, label_2, sampling_rate, output_path):
    rows = []
    for label, results in [(label_1, results_1), (label_2, results_2)]:
        for sl, m in sorted(results.items()):
            rows.append(
                {
                    "experiment": label,
                    "split_length": sl,
                    "split_seconds": sl / sampling_rate,
                    "snr_mean": m["snr_mean"],
                    "snr_ci_low": m["snr_ci"][0],
                    "snr_ci_high": m["snr_ci"][1],
                    "rmse_mean": m["rmse_mean"],
                    "rmse_ci_low": m["rmse_ci"][0],
                    "rmse_ci_high": m["rmse_ci"][1],
                    "pcc_mean": m["pcc_mean"],
                    "pcc_ci_low": m["pcc_ci"][0],
                    "pcc_ci_high": m["pcc_ci"][1],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def main(exp_1, exp_2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load configs
    model_configs_1 = get_configs(f"experiments/{exp_1}/model_configs")
    model_configs_2 = get_configs(f"experiments/{exp_2}/model_configs")
    data_config = get_configs(f"experiments/{exp_1}/data_configs")[0]

    # Filter out stage-2 models (they require two-stage inference)
    model_configs_1 = [
        c for c in model_configs_1 if not c["model"].get("is_stage_2", False)
    ]
    model_configs_2 = [
        c for c in model_configs_2 if not c["model"].get("is_stage_2", False)
    ]

    # Assert one model type per experiment and extract names for plotting
    model_name_1 = _get_experiment_model_name(model_configs_1)
    model_name_2 = _get_experiment_model_name(model_configs_2)
    print(f"Experiment models: {model_name_1} vs {model_name_2}")

    sampling_rate = data_config["simulation_params"]["sampling_rate"]
    record_length = get_record_length(data_config)
    print(
        f"Record length: {record_length} samples ({record_length / sampling_rate:.1f}s)"
    )

    # Load eval records once — both experiments see the same noisy signals
    print("Loading evaluation records...")
    records = load_eval_records(data_config)
    print(f"Loaded {len(records)} records")

    # Evaluate both experiments
    print(f"\nEvaluating {exp_1}:")
    results_1 = evaluate_experiment(
        exp_1, model_configs_1, data_config, records, device
    )

    print(f"\nEvaluating {exp_2}:")
    results_2 = evaluate_experiment(
        exp_2, model_configs_2, data_config, records, device
    )

    # Save outputs
    output_dir = f"outputs/record_level_eval/{exp_1}_vs_{exp_2}"
    os.makedirs(output_dir, exist_ok=True)

    save_results_csv(
        results_1,
        results_2,
        exp_1,
        exp_2,
        sampling_rate,
        f"{output_dir}/results.csv",
    )
    plot_comparison(
        results_1,
        results_2,
        model_name_1,
        model_name_2,
        sampling_rate,
        f"{output_dir}/snr_comparison.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record-level evaluation comparing two experiments across split lengths"
    )
    parser.add_argument(
        "--exp_1",
        required=True,
        help="First experiment name (e.g., length_syn)",
    )
    parser.add_argument(
        "--exp_2",
        required=True,
        help="Second experiment name (e.g., length_syn_mamba_no_comp)",
    )
    args = parser.parse_args()
    main(args.exp_1, args.exp_2)
