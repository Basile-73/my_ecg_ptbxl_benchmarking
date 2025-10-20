import pandas as pd
import os
import sys
import numpy as np
import ast

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from mycode.eda import ptbxl_eda


signal_stats_path = 'eda/output/data/signal_statistics.csv'
output_folder = 'eda/output/test/'

if __name__ == "__main__":
    stats_df = pd.read_csv(signal_stats_path)
    # Keep as lists instead of converting to numpy arrays to avoid dimensionality issues with boxplot
    stats_df['means'] = stats_df['means'].apply(lambda x: ast.literal_eval(x))
    stats_df['stds'] = stats_df['stds'].apply(lambda x: ast.literal_eval(x))

    #plot simple boxplots of means and stds per lead
    ptbxl_eda.plot_signal_statistics(stats_df, output_folder)
