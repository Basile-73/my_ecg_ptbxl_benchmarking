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
import matplotlib.font_manager as fm
from matplotlib.colors import Normalize
from pathlib import Path

# Register CMU Serif font
_FONT_DIR = Path(__file__).resolve().parent.parent.parent / "fonts" / "cm-unicode-0.7.0"
for _ttf in _FONT_DIR.glob("*.ttf"):
    fm.fontManager.addfont(str(_ttf))
plt.rcParams["font.family"] = "CMU Serif"

# ──────────────────────────────────────────────────────────────
# PATHS  (edit as needed)
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

RESULTS_CSV = REPO_ROOT / "mycode/denoising/output/report_strong_ls/downstream_results/exp0/per_class_roc_results_exp0.csv"
SCP_STATEMENTS = REPO_ROOT / "data/physionet.org/files/ptb-xl/1.0.3/scp_statements.csv"
Y_TEST_PATH = REPO_ROOT / "new_code/classification/output2/exp0/data/y_test.npy"
MLB_PATH = REPO_ROOT / "new_code/classification/output2/exp0/data/mlb.pkl"

OUTPUT_DIR = REPO_ROOT / "new_code/visualisation/output/trees/report_strong"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OMIT_TITLE = True

# ──────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ──────────────────────────────────────────────────────────────
LEAD_SENSITIVE_MODELS = ["mamba1_3blocks_ls", "mamba1_3blocks"]
NON_LS_EQUIVALENTS = {"mamba1_3blocks_ls": "mamba1_3blocks"}
IMUNET_EQUIVALENTS = {"mamba1_3blocks_ls": "unet", "mamba1_3blocks": "unet"}#"drnet_mamba1_3blocks_ls": "drnet_unet"} # !todo rename symbol
CLASSIFIERS = ["fastai_resnet1d_wang", "fastai_inception1d"]

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
    "fastai_resnet1d_wang": "ResNet1D-Wang",
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
               sample_counts, auc_lookup, color_lookup=None, min_samples=0):
    """Build a nested structure for visualisation.

    Parameters
    ----------
    auc_lookup    : callable(diagnosis) -> float   value shown in the label
    color_lookup  : callable(diagnosis) -> float   value used for colouring
                    (optional – defaults to auc_lookup)
    min_samples   : int   minimum positive test samples to include a diagnosis

    Returns
    -------
    tree : list of dicts, one per diagnostic_class
        Each dict has keys: name, auc, count, color, children (subclasses)
        If color_lookup is given, each node also has a "color_value" key.
    """
    if color_lookup is None:
        color_lookup = auc_lookup

    has_separate_color = color_lookup is not auc_lookup

    # Group diagnoses by subclass, subclasses by class
    # Only include diagnoses that appear in our AUC data (and meet min_samples)
    diag_names = [d for d in diag_to_class
                  if not np.isnan(auc_lookup(d)) and sample_counts.get(d, 0) >= min_samples]

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
        dc_color_vals = []
        dc_weights = []
        sc_nodes = []
        for sc in sorted(class_subclasses[dc]):
            sc_count = 0
            sc_auc_vals = []
            sc_color_vals = []
            sc_weights = []
            diag_nodes = []
            for d in sorted(subclass_diags[sc]):
                auc_val = auc_lookup(d)
                color_val = color_lookup(d)
                cnt = sample_counts.get(d, 0)
                node = {
                    "name": d,
                    "auc": auc_val,
                    "count": cnt,
                    "color": CLASS_COLORS.get(dc, "#333333"),
                }
                if has_separate_color:
                    node["color_value"] = color_val
                diag_nodes.append(node)
                sc_count += cnt
                sc_auc_vals.append(auc_val)
                sc_color_vals.append(color_val if not np.isnan(color_val) else 0.0)
                sc_weights.append(cnt)

            # Weighted average for subclass
            if sum(sc_weights) > 0:
                sc_auc = np.average(sc_auc_vals, weights=sc_weights)
                sc_color = np.average(sc_color_vals, weights=sc_weights)
            else:
                sc_auc = np.mean(sc_auc_vals) if sc_auc_vals else np.nan
                sc_color = np.mean(sc_color_vals) if sc_color_vals else np.nan
            dc_count += sc_count
            dc_auc_vals.extend(sc_auc_vals)
            dc_color_vals.extend(sc_color_vals)
            dc_weights.extend(sc_weights)

            sc_node = {
                "name": sc,
                "auc": sc_auc,
                "count": sc_count,
                "color": CLASS_COLORS.get(dc, "#333333"),
                "children": diag_nodes,
            }
            if has_separate_color:
                sc_node["color_value"] = sc_color
            sc_nodes.append(sc_node)

        # Weighted average for class
        if sum(dc_weights) > 0:
            dc_auc = np.average(dc_auc_vals, weights=dc_weights)
            dc_color = np.average(dc_color_vals, weights=dc_weights)
        else:
            dc_auc = np.mean(dc_auc_vals) if dc_auc_vals else np.nan
            dc_color = np.mean(dc_color_vals) if dc_color_vals else np.nan
        dc_node = {
            "name": dc,
            "auc": dc_auc,
            "count": dc_count,
            "color": CLASS_COLORS.get(dc, "#333333"),
            "children": sc_nodes,
        }
        if has_separate_color:
            dc_node["color_value"] = dc_color
        tree.append(dc_node)

    return tree


# ──────────────────────────────────────────────────────────────
# DRAWING
# ──────────────────────────────────────────────────────────────
MARKER_SIZE_LEAF = 672       # scatter marker size in points² (always round)
MARKER_SIZE_SUBCLASS = 672
MARKER_SIZE_CLASS = 672
LEVEL_X = [0.10, 0.2, 0.50]  # x positions of the 3 levels (closer together)
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
        Uses the actual data range [vmin, vmax] for better contrast.
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
        # Use the class base_color, darken proportional to AUC
        # Rescale to actual data range so colours spread across the full gradient
        from matplotlib.colors import to_rgba
        r, g, b, _ = to_rgba(base_color)
        data_range = vmax - vmin
        if data_range > 0:
            t = max(0, min(1, (auc - vmin) / data_range))
        else:
            t = 0.5
        # t=0 (lowest AUC) -> very light pastel, t=1 (highest AUC) -> full dark
        # Light end:  base*0.3 + white*0.7  (pastel)
        # Dark end:   base*1.0 + black*0.3  (rich dark)
        lightness = 1.0 - t
        r2 = r * (0.3 + 0.7 * t) * (0.7 + 0.3 * t) + lightness * 0.5
        g2 = g * (0.3 + 0.7 * t) * (0.7 + 0.3 * t) + lightness * 0.5
        b2 = b * (0.3 + 0.7 * t) * (0.7 + 0.3 * t) + lightness * 0.5
        return (min(r2, 1), min(g2, 1), min(b2, 1))


def _node_color_value(node):
    """Return the value used for colouring (color_value if present, else auc)."""
    return node.get("color_value", node["auc"])


def draw_tree(tree, title, save_path, is_delta=False):
    """Draw the 3-level tree to a figure and save it.

    If tree nodes contain a "color_value" key, that value drives the colour
    (using the delta green/red scheme) while "auc" is shown in the labels.
    """
    # Compute total leaves for vertical spacing
    n_leaves = _count_leaves(tree)
    if n_leaves == 0:
        print(f"  Skipping {title}: no data")
        return

    # Check whether nodes carry a separate color_value
    has_color_value = "color_value" in tree[0]
    color_is_delta = is_delta or has_color_value

    # Collect all colour-driving values for normalization
    all_aucs = []
    for dc in tree:
        all_aucs.append(_node_color_value(dc))
        for sc in dc["children"]:
            all_aucs.append(_node_color_value(sc))
            for d in sc["children"]:
                all_aucs.append(_node_color_value(d))
    all_aucs = [a for a in all_aucs if not np.isnan(a)]
    vmin = min(all_aucs) if all_aucs else -0.1
    vmax = max(all_aucs) if all_aucs else 0.1

    fig_height = max(10, n_leaves * 0.55)
    fig, ax = plt.subplots(figsize=(22, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if not OMIT_TITLE:
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
            ax.plot([dc_x, sc_x], [dc_y, sc_y], color="#cccccc", linewidth=1.6, zorder=1)

            for d in sc["children"]:
                d_x = LEVEL_X[2]
                d_y = node_positions[("diag", d["name"])]
                # Hide redundant child (single child with same name as parent subclass)
                is_redundant = len(sc["children"]) == 1 and d["name"] == sc["name"]
                node_alpha = 0.0 if is_redundant else 1.0
                # Line: subclass -> diag
                ax.plot([sc_x, d_x], [sc_y, d_y], color="#cccccc", linewidth=1.6, zorder=1, alpha=node_alpha)

                # Draw diagnosis node (scatter = always round)
                fill = _get_node_color(d["color"], _node_color_value(d), color_is_delta, vmin, vmax)
                ax.scatter(d_x, d_y, s=MARKER_SIZE_LEAF, c=[fill], edgecolors="black",
                           linewidths=1.2, zorder=3, clip_on=False, alpha=node_alpha)
                if not is_redundant:
                    label = _format_label(d["name"], d["auc"], d["count"], is_delta)
                    ax.text(d_x + TEXT_PAD, d_y, label, va="center", fontsize=fontsize_label, zorder=4)

            # Draw subclass node
            fill = _get_node_color(sc["color"], _node_color_value(sc), color_is_delta, vmin, vmax)
            ax.scatter(sc_x, sc_y, s=MARKER_SIZE_SUBCLASS, c=[fill], edgecolors="black",
                       linewidths=1.5, zorder=3, clip_on=False)
            label = _format_label(sc["name"], sc["auc"], sc["count"], is_delta)
            ax.text(sc_x + TEXT_PAD, sc_y, label, va="center", fontsize=fontsize_label + 1,
                    fontweight="bold", zorder=4)

        # Draw class node
        fill = _get_node_color(dc["color"], _node_color_value(dc), color_is_delta, vmin, vmax)
        ax.scatter(dc_x, dc_y, s=MARKER_SIZE_CLASS, c=[fill], edgecolors="black",
                   linewidths=2, zorder=3, clip_on=False)
        label = _format_label(dc["name"], dc["auc"], dc["count"], is_delta)
        ax.text(dc_x - TEXT_PAD, dc_y, label, va="center", ha="right", fontsize=fontsize_label + 2,
                fontweight="bold", zorder=4)

    # Add a colorbar legend for delta / hybrid trees
    if color_is_delta:
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.colorbar as mcolorbar

        # Build a red-white-green colormap matching _get_node_color logic
        cmap_colors = []
        n_steps = 256
        for i in range(n_steps):
            val = vmin + (vmax - vmin) * i / (n_steps - 1)
            intensity = _log_intensity(val)
            if val >= 0:
                cr = 1.0 - intensity * 1.0
                cg = 1.0 - intensity * 0.45
                cb = 1.0 - intensity * 1.0
            else:
                cr = 1.0 - intensity * 0.2
                cg = 1.0 - intensity * 1.0
                cb = 1.0 - intensity * 1.0
            cmap_colors.append((max(cr, 0), max(cg, 0), max(cb, 0)))
        cmap = LinearSegmentedColormap.from_list("delta", cmap_colors, N=n_steps)
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Place colorbar at the bottom of the figure
        cbar_ax = fig.add_axes([0.25, -0.02, 0.5, 0.012])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        label_text = "Improvement over noisy" if has_color_value else "AUC delta"
        cbar.set_label(label_text, fontsize=fontsize_label)
        cbar.ax.tick_params(labelsize=fontsize_label - 2)

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

    MIN_SAMPLES_PRUNED = 5

    for model in LEAD_SENSITIVE_MODELS:
        non_ls = NON_LS_EQUIVALENTS.get(model)
        imunet = IMUNET_EQUIVALENTS.get(model)
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

            # Tree 2: Absolute AUC - Noisy AUC
            def auc_minus_noisy(d):
                if d not in diagnostic_labels:
                    return np.nan
                a = get_auc(df, model, classifier, d)
                b = get_auc(df, "noisy", classifier, d)
                if np.isnan(a) or np.isnan(b):
                    return np.nan
                return a - b

            # Tree 3: Absolute AUC - Non-lead-sensitive equivalent
            def auc_minus_non_ls(d):
                if d not in diagnostic_labels:
                    return np.nan
                a = get_auc(df, model, classifier, d)
                b = get_auc(df, non_ls, classifier, d)
                if np.isnan(a) or np.isnan(b):
                    return np.nan
                return a - b

            # Tree 4: Absolute AUC - IMUNet equivalent
            def auc_minus_imunet(d):
                if d not in diagnostic_labels:
                    return np.nan
                a = get_auc(df, model, classifier, d)
                b = get_auc(df, imunet, classifier, d)
                if np.isnan(a) or np.isnan(b):
                    return np.nan
                return a - b

            # Define all tree specs: (number, title, path, is_delta, auc_lookup, color_lookup, needs)
            tree_specs = [
                ("1", f"Absolute AUC - {model_display} / {clf_display}",
                 "1_absolute_auc", False, auc_absolute, None, None),
                ("2", f"AUC Improvement over Noisy - {model_display} / {clf_display}",
                 "2_vs_noisy", True, auc_minus_noisy, None, None),
                ("3", f"AUC Improvement over {non_ls} - {model_display} / {clf_display}",
                 "3_vs_non_ls", True, auc_minus_non_ls, None, non_ls),
                ("4", f"AUC Improvement over {imunet} - {model_display} / {clf_display}",
                 "4_vs_imunet", True, auc_minus_imunet, None, imunet),
                ("5", f"Absolute AUC (coloured by improvement over noisy) - {model_display} / {clf_display}",
                 "5_absolute_colored_by_noisy", False, auc_absolute, auc_minus_noisy, None),
            ]

            for num, title, suffix, is_delta, auc_fn, color_fn, needs in tree_specs:
                if needs is not None and not needs:
                    continue

                bt_kwargs = dict(
                    diag_to_subclass=diag_to_subclass, diag_to_class=diag_to_class,
                    subclass_to_class=subclass_to_class, all_diag_classes=all_diag_classes,
                    sample_counts=sample_counts, auc_lookup=auc_fn, color_lookup=color_fn,
                )

                # Full tree
                tree = build_tree(**bt_kwargs)
                draw_tree(tree, title, OUTPUT_DIR / f"{prefix}__{suffix}.png", is_delta=is_delta)

                # Pruned tree (min_samples filter)
                tree_pruned = build_tree(**bt_kwargs, min_samples=MIN_SAMPLES_PRUNED)
                draw_tree(tree_pruned,
                          f"{title} (≥{MIN_SAMPLES_PRUNED} samples)",
                          OUTPUT_DIR / f"{prefix}__{suffix}_pruned.png",
                          is_delta=is_delta)

    print(f"\nAll trees saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
