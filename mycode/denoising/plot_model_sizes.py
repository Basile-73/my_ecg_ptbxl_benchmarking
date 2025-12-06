import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# TODO: use same colors and order as in qui plot
data = {
    "Model": [
        "mecge_phase",
        "imunet_mamba_bottleneck",
        "imunet_mamba_early",
        "imunet",
        "drnet_imunet",
        "unet",
        "drnet_unet"
    ],
    "Parameters": [819561, 526781, 323386, 277124, 505964, 226737, 455577],
    "EarlyStoppingEpochs": [142, 67, 98, 46, 45, 41, 58]
}


color_map = {
        'noisy_input': '#808080',  # Grey (baseline)
        'fcn': '#aec7e8',         # Light blue (Stage1)
        'drnet_fcn': '#1f77b4',   # Dark blue (Stage2)
        'unet': '#ff9896',        # Light red (Stage1)
        'drnet_unet': '#d62728',  # Dark red (Stage2)
        'imunet': '#98df8a',      # Light green (Stage1)
        'drnet_imunet': '#2ca02c', # Dark green (Stage2)
        'imunet_origin': '#9467bd',    # Purple
        'mecge_phase': '#C91CB5',
        'imunet_mamba_bn': '#ff7f0e',  # Orange
        'imunet_mamba_bottleneck': '#1C8AC9',  # Orange
        'imunet_mamba_up': '#17becf',  # Cyan/Teal
        'imunet_mamba_early': '#391CC9', # Magenta/Pink
        'imunet_mamba_late': '#bcbd22',  # Yellow-green
    }

df = pd.DataFrame(data)
order = [k for k in color_map if k in df.Model.values]
df = df.set_index("Model").loc[order].reset_index()
colors = [color_map[m] for m in df["Model"]]

x = np.arange(len(df))
w = 0.4

grey = "#888888"

fig, ax1 = plt.subplots(figsize=(10,3))
ax2 = ax1.twinx()

ax1.bar(x - w/2, df["Parameters"], width=w, color=colors, label="Parameters")
ax2.bar(x + w/2, df["EarlyStoppingEpochs"], width=w, color=grey, label="Early Stopping")

ax1.set_ylabel("Parameters", rotation=90)
ax2.set_ylabel("Early Stopping Epochs", rotation=90)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Combine legends from both axes
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)

ax1.set_xticks(x)
ax1.set_xticklabels(df["Model"], rotation=45, ha='right')

plt.tight_layout()  # now fits everything cleanly

plt.savefig("grouped_plot.png", dpi=300)
plt.close()
