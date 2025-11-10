import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Model": [
        "fcn (Qui)", "imunet (Qui)", "imunet_mamba_bn",
        "imunet_mamba_early", "imunet_mamba_late",
        "imunet_mamba_up", "mecge_phase (Hung)", "unet (Qui)"
    ],
    "Parameters": [193457, 277124, 328253, 278554, 285117, 309885, 819561, 226737]
}

color_map = {
    'fcn (Qui)': '#6baed6',
    'unet (Qui)': '#fc8d62',
    'imunet (Qui)': '#66c2a5',
    'imunet_mamba_bn': '#ff7f0e',
    'imunet_mamba_up': '#17becf',
    'imunet_mamba_early': '#e377c2',
    'imunet_mamba_late': '#bcbd22',
    'mecge_phase (Hung)': '#c5b0d5'
}

df = pd.DataFrame(data)
colors = [color_map[m] for m in df["Model"]]

plt.figure(figsize=(8,4))
plt.bar(df["Model"], df["Parameters"], color=colors)
plt.xticks(rotation=45, ha='right')
plt.title("Model Parameter Counts")
plt.tight_layout()
plt.savefig("model_sizes.png", dpi=300)
plt.close()
