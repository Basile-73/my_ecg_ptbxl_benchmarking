import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from maps import COLOR_MAP, NAME_MAP, plot_font_sizes

choice = 'european' # 'european', 'sinus', 'ptb-xl', 'synthetic'
legend = False
save_figure = True
save_table = False


models = [
    'chiang_dae',
    'ant_drnn',
    'mecge_phase',
    'unet',
    'drnet_unet',
    'imunet',
    'drnet_imunet',
    'mamba1_3blocks',
    #'mamba2_3blocks',
    'drnet_mamba1_3blocks',
    #'drnet_mamba2_3blocks',

    # 'mamba1_3blocks_ptb_l0',
    # 'mamba1_3blocks_ptb_l1',
    # 'mamba1_3blocks_ptb_l2',
    # 'mamba1_3blocks_ptb_l3',
    # 'mamba1_3blocks_ptb_l4',
    # 'mamba1_3blocks_ptb_l5',
    # 'mamba1_3blocks_ptb_l6',
    # 'mamba1_3blocks_ptb_l7',
    # 'mamba1_3blocks_ptb_l8',
    # 'mamba1_3blocks_ptb_l9',
    # 'mamba1_3blocks_ptb_l10',
    # 'mamba1_3blocks_ptb_l11',
    # 'mamba1_3blocks_ptb_all',
    # 'drnet_mamba1_3blocks_l0',
    # 'drnet_mamba1_3blocks_l1',
    # 'drnet_mamba1_3blocks_l2',
    # 'drnet_mamba1_3blocks_l3',
    # 'drnet_mamba1_3blocks_l4',
    # 'drnet_mamba1_3blocks_l5',
    # 'drnet_mamba1_3blocks_l6',
    # 'drnet_mamba1_3blocks_l7',
    # 'drnet_mamba1_3blocks_l8',
    # 'drnet_mamba1_3blocks_l9',
    # 'drnet_mamba1_3blocks_l10',
    # 'drnet_mamba1_3blocks_l11',
    # 'drnet_mamba1_3blocks_all',

]

datasets = {
    'european': 'reproduce_eu/european_st_t',
    'sinus': 'all_models_sinus/mitbih_sinus/14400',
    'ptb-xl': 'all_models_ptb_xl/3600',
    'synthetic': 'reproduce_syn/synthetic'
}

dfs = []
################################################################################
# Add to dfs
################################################################################
# for model in models:
#     if choice != 'synthetic':
#         df = pd.read_csv(Path(f'../outputs/{datasets[choice]}_{model}_1/results.csv'))
#     else:
#         df = pd.read_csv(Path(f'../outputs/AAA_performance_comparison/{model}.csv')) # This for Synthetic
#     df["model"] = model
#     dfs.append(df)

for model in models:
    if model == 'mecge_phase':
        model_name = f'1800_{model}'
    else:
        model_name = f'14400_{model}'
    df = pd.read_csv(f'../outputs/{datasets[choice]}/{model_name}/results.csv')
    df['model'] = model
    dfs.append(df)

all_results = pd.concat(dfs, ignore_index=True)

################################################################################
# Compute Table
################################################################################

if save_table:
    table_save_path = Path(f'../outputs/AAA_tabels/{choice}.csv')
    all_results.to_csv(table_save_path, index=False)
    print(f"Results summary table saved to {table_save_path}")
out = all_results

################################################################################
# Compute Figure
################################################################################

fig, axes = plt.subplots(1, 2, figsize=(len(models)*1, 5))
#fig.suptitle("Model Performance", fontsize=plot_font_sizes['title'], fontweight="bold")

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
            color=COLOR_MAP.get(model, "#C9C9C9"),
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
            ha="center", va="bottom", fontsize=plot_font_sizes['value_labels'], rotation=90
        )

    ymin = (means - ylow).min() * 0.95
    ymax = (means + yhigh).max() + text_pad * 17
    if (metric == "RMSE") and (choice == 'european'):
        ax.set_ylim(ymin, 0.8)
    else:
        ax.set_ylim(ymin, ymax)

    # Add dotted horizontal line for best performing model
    best_value = means.min() if metric == "RMSE" else means.max()
    ax.axhline(y=best_value, color='grey', linestyle=':', linewidth=2, alpha=0.7, label='Best')

    #ax.set_title(f"{metric} (95% CI)", fontweight="bold")
    ax.set_ylabel(metric, fontweight="bold", fontsize=plot_font_sizes['axis_labels'])
    ax.set_xticks(range(len(models)))
    #ax.set_xticklabels(models, rotation=90)
    ax.set_xticks([])
    ax.tick_params(axis='both', which='major', labelsize=plot_font_sizes['ticks'])
    ax.grid(True, axis="y", alpha=0.3)

# Create legend labels with "(ours)" for mamba models
legend_labels = [f"{NAME_MAP.get(m, m)} (ours)" if 'mamba' in m and 'drnet' not in m else
                 f"{NAME_MAP.get(m, m)} (ours)" if m == 'drnet_mamba1_3blocks' or m == 'drnet_mamba2_3blocks' else
                 NAME_MAP.get(m, m) for m in models]

if legend == True:
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.54, -0.1), ncol=len(models)//2, frameon=True, fontsize=plot_font_sizes['legend'])
plt.tight_layout(rect=[0, 0.05, 1, 1])

if save_figure:
    save_path = Path(f'../outputs/plots/{choice}_{legend}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")

plt.show()
