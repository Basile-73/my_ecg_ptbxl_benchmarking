import pandas as pd
from pathlib import Path
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from maps import COLOR_MAP, NAME_MAP, plot_font_sizes

# Register CMU Serif font
_FONT_DIR = Path(__file__).resolve().parent.parent.parent / "fonts" / "cm-unicode-0.7.0"
for _ttf in _FONT_DIR.glob("*.ttf"):
    fm.fontManager.addfont(str(_ttf))
plt.rcParams["font.family"] = "CMU Serif"

# Scale all font sizes by 1.2
font_sizes = {k: v * 1.44 for k, v in plot_font_sizes.items()}

save_figure = True

# (dataset_label, prefix) pairs
datasets = [
    ('European ST-T', 'eu'),
    ('Synthetic',     'syn'),
    ('PTB-XL',        'ptb'),
]

# (suffix, y_label) pairs
metrics = [
    ('test',  'Test SNR (dB)'),
    ('train', 'Training Loss'),
]


def get_model_name(col):
    """Parse model name from column like 'model_name - test/SNR'."""
    return col.split(' - ')[0]


for dataset_label, prefix in datasets:
    for suffix, y_label in metrics:
        csv_path = Path(f'../outputs/training_data/{prefix}_{suffix}.csv')
        df = pd.read_csv(csv_path)

        # Extract model columns: skip MIN/MAX, keep only mean columns
        metric_cols = [
            c for c in df.columns
            if c != 'Step' and '__MIN' not in c and '__MAX' not in c
        ]

        # Split into drnet (stage 2) and non-drnet (stage 1) groups
        drnet_cols = [c for c in metric_cols if 'drnet' in get_model_name(c)]
        non_drnet_cols = [c for c in metric_cols if 'drnet' not in get_model_name(c)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax, cols, title in zip(
            axes,
            [non_drnet_cols, drnet_cols],
            ['Stage 1 models', 'Stage 2 models (DRNET)'],
        ):
            for col in cols:
                model = get_model_name(col)
                ax.plot(
                    df['Step'],
                    df[col],
                    label=NAME_MAP.get(model, model),
                    color=COLOR_MAP.get(model, '#C9C9C9'),
                    linewidth=1.5,
                )

            ax.set_title(title, fontsize=font_sizes['title'], fontweight='bold')
            ax.set_xlabel('Step', fontsize=font_sizes['axis_labels'])
            ax.set_ylabel(y_label, fontsize=font_sizes['axis_labels'])
            ax.tick_params(axis='both', labelsize=font_sizes['ticks'])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=font_sizes['legend'])

        plt.tight_layout()

        if save_figure:
            output_name = f'training_{suffix}_{prefix}'
            save_path = Path(f'../outputs/plots/{output_name}.png')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

# --- Compression comparison plot ---
compression_csv = Path('../outputs/training_data/compression.csv')
df_comp = pd.read_csv(compression_csv)

metric_cols = [
    c for c in df_comp.columns
    if c != 'Step' and '__MIN' not in c and '__MAX' not in c
]

fig, ax = plt.subplots(figsize=(8, 5))
for col in metric_cols:
    model = get_model_name(col)
    ax.plot(
        df_comp['Step'],
        df_comp[col],
        label=NAME_MAP.get(model, model),
        color=COLOR_MAP.get(model, '#C9C9C9'),
        linewidth=1.5,
    )

ax.set_title('Compression vs No Compression', fontsize=font_sizes['title'], fontweight='bold')
ax.set_xlabel('Step', fontsize=font_sizes['axis_labels'])
ax.set_ylabel('Test RMSE', fontsize=font_sizes['axis_labels'])
ax.tick_params(axis='both', labelsize=font_sizes['ticks'])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=font_sizes['legend'])
plt.tight_layout()

if save_figure:
    save_path = Path('../outputs/plots/compression_rmse.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")

plt.show()
