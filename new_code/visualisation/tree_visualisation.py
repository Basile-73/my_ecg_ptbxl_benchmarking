"""
Tree visualisation of per-class AUC results following the PTB-XL diagnostic hierarchy.

Generates 4 tree types for each of the 2 lead-sensitive models x 2 classifiers:
  1. Absolute AUC
  2. Absolute AUC - Noisy AUC
  3. Absolute AUC - AUC of the non-lead-sensitive equivalent
  4. Absolute AUC - AUC of the corresponding IMUNet

Usage:
    python tree_visualisation.py
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# PATHS  (edit as needed)
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

RESULTS_CSV = REPO_ROOT / "mycode/denoising/output/test_ls/downstream_results/exp0/per_class_roc_results_exp0.csv"
SCP_STATEMENTS = REPO_ROOT / "data/physionet.org/files/ptb-xl/1.0.3/scp_statements.csv"
Y_TEST_PATH = REPO_ROOT / "new_code/classification/output2/exp0/data/y_test.npy"
MLB_PATH = REPO_ROOT / "new_code/classification/output2/exp0/data/mlb.pkl"

OUTPUT_DIR = REPO_ROOT / "new_code/visualisation/output/trees"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ──────────────────────────────────────────────────────────────
LEAD_SENSITIVE_MODELS = ["mamba1_3blocks_ls", "drnet_mamba1_3blocks_ls"]
NON_LS_EQUIVALENTS = {"mamba1_3blocks_ls": "mamba1_3blocks", "drnet_mamba1_3blocks_ls": "drnet_mamba1_3blocks"}
IMUNET_EQUIVALENTS = {"mamba1_3blocks_ls": "imunet", "drnet_mamba1_3blocks_ls": "drnet_imunet"}
CLASSIFIERS = ["fastai_xresnet1d101", "fastai_inception1d"]

# ──────────────────────────────────────────────────────────────
# COLOUR PALETTE  – one colour per diagnostic_class
# ──────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "STTC": "#808080",   # grey
    "MI":   "#4a90d9",   # blue
    "HYP":  "#9b59b6",   # purple
    "CD":   "#e67e22",   # orange
    "NORM": "#27ae60",   # green
}

MODEL_DISPLAY_NAMES = {
    "mamba1_3blocks_ls": "Mamba1-3B (Lead Aware)",
    "drnet_mamba1_3blocks_ls": "DRNET Mamba1-3B (Lead Aware)",
}

CLASSIFIER_DISPLAY_NAMES = {
    "fastai_xresnet1d101": "XResNet1D-101",
    "fastai_inception1d": "Inception1D",
}


# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────
def load_hierarchy():
    """Return the 3-level diagnostic hierarchy from scp_statements.csv.

    Returns
    -------
    diag_to_subclass : dict   diagnosis -> diagnostic_subclass
    diag_to_class    : dict   diagnosis -> diagnostic_class
    subclass_to_class: dict   diagnostic_subclass -> diagnostic_class
    all_diag_classes : list   sorted list of diagnostic_class values
    """
    scp = pd.read_csv(SCP_STATEMENTS, index_col=0)
    # Keep only diagnostic rows
    diag = scp[scp["diagnostic"] == 1.0].copy()

    diag_to_subclass = diag["diagnostic_subclass"].to_dict()
    diag_to_class = diag["diagnostic_class"].to_dict()

    # Build subclass -> class mapping
    subclass_to_class = {}
    for _, row in diag.iterrows():
        sc = row["diagnostic_subclass"]
        dc = row["diagnostic_class"]
        if pd.notna(sc) and pd.notna(dc):
            subclass_to_class[sc] = dc

    all_diag_classes = sorted(set(diag_to_class.values()))
    return diag_to_subclass, diag_to_class, subclass_to_class, all_diag_classes


def load_sample_counts():
    """Return dict: diagnosis_name -> number of positive test samples."""
    y_test = np.load(Y_TEST_PATH, allow_pickle=True)
    with open(MLB_PATH, "rb") as f:
        mlb = pickle.load(f)
    counts = y_test.sum(axis=0)
    return {name: int(c) for name, c in zip(mlb.classes_, counts)}


def load_auc_table():
    """Return the full per-class ROC results DataFrame."""
    return pd.read_csv(RESULTS_CSV)


def get_auc(df, model, classifier, diagnosis):
    """Fetch the roc_auc value for a given (model, classifier, diagnosis)."""
    row = df[(df["denoising_model"] == model) & (df["classifier"] == classifier) & (df["diagnosis"] == diagnosis)]
    if len(row) == 0:
        return np.nan
    return row["roc_auc"].values[0]


# ──────────────────────────────────────────────────────────────
# HIERARCHY BUILDING
# ──────────────────────────────────────────────────────────────
def build_tree(diag_to_subclass, diag_to_class, subclass_to_class, all_diag_classes,
               sample_counts, auc_lookup):
    """Build a nested structure for visualisation.

    Parameters
    ----------
    auc_lookup : callable(diagnosis) -> float   returns AUC (or delta) for a diagnosis

    Returns
    -------
    tree : list of dicts, one per diagnostic_class
        Each dict has keys: name, auc, count, color, children (subclasses)
        Each subclass dict has: name, auc, count, color, children (diagnoses)
    """
    # Group diagnoses by subclass, subclasses by class
    # Only include diagnoses that appear in our AUC data
    diag_names = [d for d in diag_to_class if not np.isnan(auc_lookup(d))]

    # Build subclass groups
    subclass_diags = {}  # subclass -> list of diagnosis names
    for d in diag_names:
        sc = diag_to_subclass.get(d)
        if pd.isna(sc):
            continue
        subclass_diags.setdefault(sc, []).append(d)

    # Build class groups
    class_subclasses = {}  # class -> list of subclass names
    for sc in subclass_diags:
        dc = subclass_to_class.get(sc)
        if dc is None:
            continue
        class_subclasses.setdefault(dc, []).append(sc)

    tree = []
    for dc in all_diag_classes:
        if dc not in class_subclasses:
            continue
        dc_count = 0
        dc_auc_vals = []
        dc_weights = []
        sc_nodes = []
        for sc in sorted(class_subclasses[dc]):
            sc_count = 0
            sc_auc_vals = []
            sc_weights = []
            diag_nodes = []
            for d in sorted(subclass_diags[sc]):
                auc_val = auc_lookup(d)
                cnt = sample_counts.get(d, 0)
                diag_nodes.append({
                    "name": d,
                    "auc": auc_val,
                    "count": cnt,
                    "color": CLASS_COLORS.get(dc, "#333333"),
                })
                sc_count += cnt
                sc_auc_vals.append(auc_val)
                sc_weights.append(cnt)

            # Weighted average AUC for subclass
            if sum(sc_weights) > 0:
                sc_auc = np.average(sc_auc_vals, weights=sc_weights)
            else:
                sc_auc = np.mean(sc_auc_vals) if sc_auc_vals else np.nan
            dc_count += sc_count
            dc_auc_vals.extend(sc_auc_vals)
            dc_weights.extend(sc_weights)

            sc_nodes.append({
                "name": sc,
                "auc": sc_auc,
                "count": sc_count,
                "color": CLASS_COLORS.get(dc, "#333333"),
                "children": diag_nodes,
            })

        # Weighted average AUC for class
        if sum(dc_weights) > 0:
            dc_auc = np.average(dc_auc_vals, weights=dc_weights)
        else:
            dc_auc = np.mean(dc_auc_vals) if dc_auc_vals else np.nan
        tree.append({
            "name": dc,
            "auc": dc_auc,
            "count": dc_count,
            "color": CLASS_COLORS.get(dc, "#333333"),
            "children": sc_nodes,
        })

    return tree


# ──────────────────────────────────────────────────────────────
# DRAWING
# ──────────────────────────────────────────────────────────────
MARKER_SIZE_LEAF = 320       # scatter marker size in points² (always round)
MARKER_SIZE_SUBCLASS = 400
MARKER_SIZE_CLASS = 560
LEVEL_X = [0.10, 0.3, 0.60]  # x positions of the 3 levels (closer together)
Y_MARGIN = 0.02  # top/bottom margin
TEXT_PAD = 0.025  # horizontal gap between marker centre and label


def _count_leaves(tree):
    """Total number of leaf (diagnosis) nodes."""
    total = 0
    for dc_node in tree:
        for sc_node in dc_node["children"]:
            total += len(sc_node["children"])
    return total


def _format_label(name, auc, count, is_delta=False):
    """Format a node label like '[0.934] STTC (523)'."""
    if np.isnan(auc):
        auc_str = "[NaN]"
    elif is_delta:
        auc_str = f"[{auc:+.3f}]"
    else:
        auc_str = f"[{auc:.3f}]"
    return f"{auc_str} {name} ({count})"


def _log_intensity(value, threshold=0.001):
    """Map an absolute delta value to 0..1 intensity using a log scale.

    Small values near zero get low intensity (pale), large values get
    high intensity (saturated).  A threshold clips very small values
    to 0 so they stay white-ish.
    """
    abs_val = abs(value)
    if abs_val < threshold:
        return 0.0
    # log1p(x*k) gives a nice curve; k controls how fast it saturates
    k = 50.0  # tweak: higher = faster saturation for small deltas
    intensity = np.log1p(abs_val * k) / np.log1p(k)  # normalise so max ~1 when abs_val~1
    return min(intensity, 1.0)


def _get_node_color(base_color, auc, is_delta, vmin, vmax):
    """Return the fill color for a node circle.

    For absolute AUC: shade the base_color by AUC (lighter = higher).
    For delta trees: green for positive, red for negative.
        Intensity follows a log scale so outliers don't wash everything else out.
    """
    if np.isnan(auc):
        return "#cccccc"

    if is_delta:
        intensity = _log_intensity(auc)
        if auc >= 0:
            # White -> dark green   (0,0.55,0)
            r = 1.0 - intensity * 1.0
            g = 1.0 - intensity * 0.45
            b = 1.0 - intensity * 1.0
        else:
            # White -> dark red     (0.8,0,0)
            r = 1.0 - intensity * 0.2
            g = 1.0 - intensity * 1.0
            b = 1.0 - intensity * 1.0
        return (max(r, 0), max(g, 0), max(b, 0))
    else:
        # Use the class base_color with alpha proportional to AUC
        from matplotlib.colors import to_rgba
        r, g, b, _ = to_rgba(base_color)
        # Map AUC 0.5..1.0 to alpha 0.3..1.0
        alpha = 0.3 + 0.7 * max(0, min(1, (auc - 0.5) / 0.5))
        # Blend with white
        r2 = r * alpha + 1.0 * (1 - alpha)
        g2 = g * alpha + 1.0 * (1 - alpha)
        b2 = b * alpha + 1.0 * (1 - alpha)
        return (r2, g2, b2)


def draw_tree(tree, title, save_path, is_delta=False):
    """Draw the 3-level tree to a figure and save it."""
    # Compute total leaves for vertical spacing
    n_leaves = _count_leaves(tree)
    if n_leaves == 0:
        print(f"  Skipping {title}: no data")
        return

    # Collect all AUC values for color normalization
    all_aucs = []
    for dc in tree:
        all_aucs.append(dc["auc"])
        for sc in dc["children"]:
            all_aucs.append(sc["auc"])
            for d in sc["children"]:
                all_aucs.append(d["auc"])
    all_aucs = [a for a in all_aucs if not np.isnan(a)]
    vmin = min(all_aucs) if all_aucs else -0.1
    vmax = max(all_aucs) if all_aucs else 0.1

    fig_height = max(10, n_leaves * 0.55)
    fig, ax = plt.subplots(figsize=(22, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=28, fontweight="bold", pad=20)

    # Layout: assign y positions bottom-up for leaves, then propagate to parents
    y_step = (1 - 2 * Y_MARGIN) / max(n_leaves - 1, 1)
    leaf_y = Y_MARGIN

    # First pass: assign y to all leaves and compute parent y as midpoint
    node_positions = {}  # (level, name) -> y

    leaf_idx = 0
    for dc in tree:
        dc_leaf_ys = []
        for sc in dc["children"]:
            sc_leaf_ys = []
            for d in sc["children"]:
                y = 1 - (Y_MARGIN + leaf_idx * y_step)
                node_positions[("diag", d["name"])] = y
                sc_leaf_ys.append(y)
                leaf_idx += 1
            # Subclass y = midpoint of its children
            sc_y = np.mean(sc_leaf_ys)
            node_positions[("subclass", sc["name"])] = sc_y
            dc_leaf_ys.extend(sc_leaf_ys)
        # Class y = midpoint of all its leaves
        dc_y = np.mean(dc_leaf_ys)
        node_positions[("class", dc["name"])] = dc_y

    # Draw connections and nodes (scatter markers are always round)
    fontsize_label = max(18, min(22, 1000 / n_leaves))

    for dc in tree:
        dc_x = LEVEL_X[0]
        dc_y = node_positions[("class", dc["name"])]

        for sc in dc["children"]:
            sc_x = LEVEL_X[1]
            sc_y = node_positions[("subclass", sc["name"])]
            # Line: class -> subclass
            ax.plot([dc_x, sc_x], [dc_y, sc_y], color="#cccccc", linewidth=0.8, zorder=1)

            for d in sc["children"]:
                d_x = LEVEL_X[2]
                d_y = node_positions[("diag", d["name"])]
                # Line: subclass -> diag
                ax.plot([sc_x, d_x], [sc_y, d_y], color="#cccccc", linewidth=0.8, zorder=1)

                # Draw diagnosis node (scatter = always round)
                fill = _get_node_color(d["color"], d["auc"], is_delta, vmin, vmax)
                ax.scatter(d_x, d_y, s=MARKER_SIZE_LEAF, c=[fill], edgecolors=d["color"],
                           linewidths=1.2, zorder=3, clip_on=False)
                label = _format_label(d["name"], d["auc"], d["count"], is_delta)
                ax.text(d_x + TEXT_PAD, d_y, label, va="center", fontsize=fontsize_label, zorder=4)

            # Draw subclass node
            fill = _get_node_color(sc["color"], sc["auc"], is_delta, vmin, vmax)
            ax.scatter(sc_x, sc_y, s=MARKER_SIZE_SUBCLASS, c=[fill], edgecolors=sc["color"],
                       linewidths=1.5, zorder=3, clip_on=False)
            label = _format_label(sc["name"], sc["auc"], sc["count"], is_delta)
            ax.text(sc_x + TEXT_PAD, sc_y, label, va="center", fontsize=fontsize_label + 1,
                    fontweight="bold", zorder=4)

        # Draw class node
        fill = _get_node_color(dc["color"], dc["auc"], is_delta, vmin, vmax)
        ax.scatter(dc_x, dc_y, s=MARKER_SIZE_CLASS, c=[fill], edgecolors=dc["color"],
                   linewidths=2, zorder=3, clip_on=False)
        label = _format_label(dc["name"], dc["auc"], dc["count"], is_delta)
        ax.text(dc_x - TEXT_PAD, dc_y, label, va="center", ha="right", fontsize=fontsize_label + 2,
                fontweight="bold", zorder=4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    diag_to_subclass, diag_to_class, subclass_to_class, all_diag_classes = load_hierarchy()
    sample_counts = load_sample_counts()
    df = load_auc_table()

    # Only keep diagnostic labels (the 44 that have diagnostic=1.0)
    diagnostic_labels = set(diag_to_class.keys())

    for model in LEAD_SENSITIVE_MODELS:
        non_ls = NON_LS_EQUIVALENTS[model]
        imunet = IMUNET_EQUIVALENTS[model]
        model_display = MODEL_DISPLAY_NAMES.get(model, model)

        for classifier in CLASSIFIERS:
            clf_display = CLASSIFIER_DISPLAY_NAMES.get(classifier, classifier)
            prefix = f"{model}__{classifier}"
            print(f"\n=== {model_display} / {clf_display} ===")

            # Tree 1: Absolute AUC
            def auc_absolute(d):
                if d not in diagnostic_labels:
                    return np.nan
                return get_auc(df, model, classifier, d)

            tree = build_tree(diag_to_subclass, diag_to_class, subclass_to_class,
                              all_diag_classes, sample_counts, auc_absolute)
            draw_tree(tree,
                      f"Absolute AUC - {model_display} / {clf_display}",
                      OUTPUT_DIR / f"{prefix}__1_absolute_auc.png",
                      is_delta=False)

            # Tree 2: Absolute AUC - Noisy AUC
            def auc_minus_noisy(d):
                if d not in diagnostic_labels:
                    return np.nan
                a = get_auc(df, model, classifier, d)
                b = get_auc(df, "noisy", classifier, d)
                if np.isnan(a) or np.isnan(b):
                    return np.nan
                return a - b

            tree = build_tree(diag_to_subclass, diag_to_class, subclass_to_class,
                              all_diag_classes, sample_counts, auc_minus_noisy)
            draw_tree(tree,
                      f"AUC Improvement over Noisy - {model_display} / {clf_display}",
                      OUTPUT_DIR / f"{prefix}__2_vs_noisy.png",
                      is_delta=True)

            # Tree 3: Absolute AUC - Non-lead-sensitive equivalent
            def auc_minus_non_ls(d):
                if d not in diagnostic_labels:
                    return np.nan
                a = get_auc(df, model, classifier, d)
                b = get_auc(df, non_ls, classifier, d)
                if np.isnan(a) or np.isnan(b):
                    return np.nan
                return a - b

            tree = build_tree(diag_to_subclass, diag_to_class, subclass_to_class,
                              all_diag_classes, sample_counts, auc_minus_non_ls)
            draw_tree(tree,
                      f"AUC Improvement over {non_ls} - {model_display} / {clf_display}",
                      OUTPUT_DIR / f"{prefix}__3_vs_non_ls.png",
                      is_delta=True)

            # Tree 4: Absolute AUC - IMUNet equivalent
            def auc_minus_imunet(d):
                if d not in diagnostic_labels:
                    return np.nan
                a = get_auc(df, model, classifier, d)
                b = get_auc(df, imunet, classifier, d)
                if np.isnan(a) or np.isnan(b):
                    return np.nan
                return a - b

            tree = build_tree(diag_to_subclass, diag_to_class, subclass_to_class,
                              all_diag_classes, sample_counts, auc_minus_imunet)
            draw_tree(tree,
                      f"AUC Improvement over {imunet} - {model_display} / {clf_display}",
                      OUTPUT_DIR / f"{prefix}__4_vs_imunet.png",
                      is_delta=True)

    print(f"\nAll trees saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
