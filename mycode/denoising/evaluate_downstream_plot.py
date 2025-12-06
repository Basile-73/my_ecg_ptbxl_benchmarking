import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
from evaluate_downstream import plot_downstream_results


if __name__ == "__main__":
    path = os.path.join(script_dir, "output/all_models_at_360/downstream_results/downstream_classification_results.csv")
    output_dir = os.path.join(script_dir, "output/all_models_at_360/downstream_results/")
    results_df = pd.read_csv(path)
    plot_downstream_results(results_df, output_dir)
