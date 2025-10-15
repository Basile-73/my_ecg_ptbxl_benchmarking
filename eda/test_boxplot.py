"""
Quick test to verify the updated signal_statistics.png plot with boxplots.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))

# Create mock data similar to what the EDA produces
LEAD_NAMES = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Mock statistics data
np.random.seed(42)
stats_df = pd.DataFrame({
    'lead': LEAD_NAMES,
    'mean_signal_mean': np.random.randn(12) * 0.002,
    'std_signal_mean': np.random.rand(12) * 0.1 + 0.02,
    'mean_signal_std': np.random.rand(12) * 0.15 + 0.1,
    'std_signal_std': np.random.rand(12) * 0.1 + 0.05,
})

print("Mock data created:")
print(stats_df)

# Test the plotting function
output_folder = '/local/home/bamorel/my_ecg_ptbxl_benchmarking/eda/output'
plots_folder = os.path.join(output_folder, 'plots')
os.makedirs(plots_folder, exist_ok=True)

print("\nGenerating test plot...")

# Create figure with 2 subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Boxplot of signal means per lead
mean_data = [stats_df['mean_signal_mean'].values, stats_df['std_signal_mean'].values]
bp1 = axes[0].boxplot(mean_data, labels=['Mean', 'Std'], patch_artist=True,
                      widths=0.6, showmeans=True,
                      boxprops=dict(facecolor='steelblue', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2),
                      meanprops=dict(marker='D', markerfacecolor='green', markersize=8))
axes[0].set_title('Distribution of Signal Means Across Leads', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Value', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_xlabel('Statistic Type', fontsize=12)

# Subplot 2: Boxplot of signal stds per lead
std_data = [stats_df['mean_signal_std'].values, stats_df['std_signal_std'].values]
bp2 = axes[1].boxplot(std_data, labels=['Mean', 'Std'], patch_artist=True,
                      widths=0.6, showmeans=True,
                      boxprops=dict(facecolor='coral', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2),
                      meanprops=dict(marker='D', markerfacecolor='green', markersize=8))
axes[1].set_title('Distribution of Signal Stds Across Leads', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Value', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_xlabel('Statistic Type', fontsize=12)

plt.tight_layout()
test_path = os.path.join(plots_folder, 'signal_statistics_test.png')
plt.savefig(test_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nTest plot saved to: {test_path}")
print("\nBoxplot interpretation:")
print("- Red line: Median")
print("- Green diamond: Mean")
print("- Box: Interquartile range (IQR)")
print("- Whiskers: Data range (typically 1.5*IQR)")
print("\nTest completed successfully!")
