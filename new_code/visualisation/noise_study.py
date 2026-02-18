import pandas as pd
import matplotlib.pyplot as plt

def plot_noise_study(csv_path="outputs/noise_study_results.csv", output_folder ='outputs/noise_study', metric='SNR'):
    """Plot noise study results: side-by-side subplots for each noise_type"""
    # Configure font sizes (increased by factor of 2)
    plt.rcParams.update({
        'font.size': 24,           # default text size
        'axes.titlesize': 28,      # title size
        'axes.labelsize': 26,      # x and y labels
        'xtick.labelsize': 22,     # x tick labels
        'ytick.labelsize': 22,     # y tick labels
        'legend.fontsize': 24,     # legend
    })

    df = pd.read_csv(csv_path)
    data = df[df['metric'] == metric]

    noise_types = data['noise_type'].unique()

    # Create individual plots for each noise type
    for noise_type in noise_types:
        # Set reduced font sizes for individual plots (factor of 2 smaller)
        plt.rcParams.update({
            'font.size': 12,           # default text size
            'axes.titlesize': 14,      # title size
            'axes.labelsize': 13,      # x and y labels
            'xtick.labelsize': 11,     # x tick labels
            'ytick.labelsize': 11,     # y tick labels
            'legend.fontsize': 12,     # legend
        })

        plt.figure(figsize=(8, 6))
        noise_data = data[data['noise_type'] == noise_type]

        # Get noise levels for custom tick positioning
        noise_levels = sorted(noise_data['noise_level'].unique())

        for model in noise_data['model_name'].unique():
            model_data = noise_data[noise_data['model_name'] == model].sort_values('noise_level')
            plt.plot(model_data['noise_level'], model_data['mean'],
                   marker='o', label=model, linewidth=2)
            plt.fill_between(model_data['noise_level'],
                          model_data['ci_low'], model_data['ci_high'], alpha=0.2)

        plt.xlabel('Noise intensity')
        plt.ylabel(metric)
        plt.title(f'{noise_type.upper()} Noise')
        plt.gca().invert_xaxis()  # Reverse the x-axis

        # Custom x-axis ticks: only show 'low' and 'high'
        plt.gca().set_xticks([min(noise_levels), max(noise_levels)])
        plt.gca().set_xticklabels(['high', 'low'])  # reversed due to inverted axis

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Save individual plot
        plt.savefig(f'{output_folder}/{noise_type}_noise_study.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Reset font sizes to original for combined plot
        plt.rcParams.update({
            'font.size': 24,           # default text size
            'axes.titlesize': 28,      # title size
            'axes.labelsize': 26,      # x and y labels
            'xtick.labelsize': 22,     # x tick labels
            'ytick.labelsize': 22,     # y tick labels
            'legend.fontsize': 24,     # legend
        })

    # Create combined figure with subplots
    fig, axes = plt.subplots(1, len(noise_types), figsize=(5*len(noise_types), 6), sharey=True)
    if len(noise_types) == 1:
        axes = [axes]

    handles, labels = None, None

    for i, noise_type in enumerate(noise_types):
        ax = axes[i]
        noise_data = data[data['noise_type'] == noise_type]

        # Get noise levels for custom tick positioning
        noise_levels = sorted(noise_data['noise_level'].unique())

        for model in noise_data['model_name'].unique():
            model_data = noise_data[noise_data['model_name'] == model].sort_values('noise_level')
            ax.plot(model_data['noise_level'], model_data['mean'],
                   marker='o', label=model, linewidth=2)
            ax.fill_between(model_data['noise_level'],
                          model_data['ci_low'], model_data['ci_high'], alpha=0.2)

        ax.set_xlabel('Noise intensity')
        ax.set_ylabel(metric)
        ax.set_title(f'{noise_type.upper()} Noise')
        ax.invert_xaxis()  # Reverse the x-axis

        # Custom x-axis ticks: only show 'low' and 'high'
        ax.set_xticks([min(noise_levels), max(noise_levels)])
        ax.set_xticklabels(['high', 'low'])  # reversed due to inverted axis

        ax.grid(True, alpha=0.3)

        # Get legend handles from first subplot
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

    # Add centered legend box at bottom
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.0),
              frameon=True, fancybox=True, shadow=True, ncol=len(labels))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)

    # Save combined figure
    plt.savefig(f'{output_folder}/noise_study_results_combined.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    results_file = 'outputs/noise_study_results_ptb_xl.csv'
    output_folder = 'outputs/noise_study/ptb_xl'
    plot_noise_study(results_file, output_folder)
