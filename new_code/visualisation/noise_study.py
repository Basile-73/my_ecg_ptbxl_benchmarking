import pandas as pd
import matplotlib.pyplot as plt

def plot_noise_study(csv_path="outputs/noise_study_results.csv", metric='SNR'):
    """Plot noise study results: side-by-side subplots for each noise_type"""
    df = pd.read_csv(csv_path)
    data = df[df['metric'] == metric]

    noise_types = data['noise_type'].unique()
    fig, axes = plt.subplots(1, len(noise_types), figsize=(5*len(noise_types), 6), sharey=True)
    if len(noise_types) == 1:
        axes = [axes]

    handles, labels = None, None

    for i, noise_type in enumerate(noise_types):
        ax = axes[i]
        noise_data = data[data['noise_type'] == noise_type]

        for model in noise_data['model_name'].unique():
            model_data = noise_data[noise_data['model_name'] == model].sort_values('noise_level')
            ax.plot(model_data['noise_level'], model_data['mean'],
                   marker='o', label=model, linewidth=2)
            ax.fill_between(model_data['noise_level'],
                          model_data['ci_low'], model_data['ci_high'], alpha=0.2)

        ax.set_xlabel('Input SNR level')
        ax.set_ylabel(metric)
        ax.set_title(f'{noise_type.upper()} Noise')
        ax.grid(True, alpha=0.3)

        # Get legend handles from first subplot
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

    # Add centered legend box at bottom
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
              frameon=True, fancybox=True, shadow=True, ncol=len(labels))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.show()
    plt.savefig('outputs/noise_study/noise_study_results.png')

if __name__ == "__main__":
    plot_noise_study()
