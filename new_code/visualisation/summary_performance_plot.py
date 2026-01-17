import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from colormap import COLOR_MAP

choice = 'ptb-xl'
legend = False
save_figure = False
save_table = False


models = [
    'ant_drnn',
    'chiang_dae',
    'mecge_phase',
    'unet',
    'drnet_unet',
    'imunet',
    'drnet_imunet',
    # 'unet_mamba',
    # 'unet_mamba_bidir',
    'mamba1_3blocks',
    'drnet_mamba1_3blocks',
    # 'mamba2_3blocks',
    # 'drnet_mamba2_3blocks',
]

datasets = {
    'european': 'all_models_europe/european_st_t/14400',
    'sinus': 'all_models_sinus/mitbih_sinus/14400',
    'ptb-xl': 'all_models_ptb_xl/3600',
    'synthetic': ''
}


dfs = []
for model in models:
    if choice != 'synthetic':
        df = pd.read_csv(Path(f'../outputs/{datasets[choice]}_{model}_1/results.csv'))
    else:
        df = pd.read_csv(Path(f'../outputs/AAA_performance_comparison/{model}.csv')) # This for Synthetic
    df["model"] = model
    dfs.append(df)

all_results = pd.concat(dfs, ignore_index=True)
if save_table:
    table_save_path = Path(f'../outputs/AAA_tabels/{choice}.csv')
    all_results.to_csv(table_save_path, index=False)
    print(f"Results summary table saved to {table_save_path}")
out = all_results

fig, axes = plt.subplots(1, 2, figsize=(len(models)*1, 5))
#fig.suptitle("Model Performance", fontsize=16, fontweight="bold")

handles = []

for ax, metric in zip(axes, ["RMSE", "SNR"]):
    d = out[out.metric == metric].set_index("model").loc[models]

    means = d["mean"].values
    ylow  = means - d["ci_low"].values
    yhigh = d["ci_high"].values - means

    ci_range = (yhigh + ylow).max()
    text_pad = ci_range * 0.3

    for i, model in enumerate(models):
        bar = ax.bar(
            i, means[i],
            yerr=[[ylow[i]], [yhigh[i]]],
            capsize=4,
            color=COLOR_MAP[model],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5
        )
        if metric == "RMSE":
            handles.append(bar[0])

        ax.text(
            i,
            means[i] + yhigh[i] + text_pad,
            f"{means[i]:.3f}" if metric == "RMSE" else f"{means[i]:.2f}",
            ha="center", va="bottom", fontsize=11, rotation=90
        )

    ymin = (means - ylow).min() * 0.95
    ymax = (means + yhigh).max() + text_pad * 13
    ax.set_ylim(ymin, ymax)

    # Add dotted horizontal line for best performing model
    best_value = means.min() if metric == "RMSE" else means.max()
    ax.axhline(y=best_value, color='grey', linestyle=':', linewidth=2, alpha=0.7, label='Best')

    #ax.set_title(f"{metric} (95% CI)", fontweight="bold")
    ax.set_ylabel(metric, fontweight="bold", fontsize=14)
    ax.set_xticks(range(len(models)))
    #ax.set_xticklabels(models, rotation=90)
    ax.set_xticks([])
    ax.grid(True, axis="y", alpha=0.3)

# Create legend labels with "(ours)" for mamba models
legend_labels = [f"{m} (ours)" if 'mamba' in m and 'drnet' not in m else
                 f"{m} (ours)" if m == 'drnet_mamba1_3blocks' or m == 'drnet_mamba2_3blocks' else
                 m for m in models]

if legend == True:
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.54, -0.1), ncol=len(models)//2, frameon=True, fontsize=11)
plt.tight_layout(rect=[0, 0.05, 1, 1])

if save_figure:
    save_path = Path(f'../outputs/AAA_plots/{choice}_{legend}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")

plt.show()
