import numpy as np
import os
import sys
from typing import Tuple
from scipy.signal import find_peaks

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from mycode.denoising.denoising_utils.preprocessing import remove_bad_labels
from mycode.denoising.denoising_utils.preprocessing import select_best_lead
from mycode.classification.utils.utils import load_dataset


SAMPLING_RATE = 500  # Hz
DATA_FOLDER = '/local/home/bamorel/my_ecg_ptbxl_benchmarking/data/ptbxl/'
OUTPUT_FOLDER = '/local/home/bamorel/my_ecg_ptbxl_benchmarking/eda/output/'



if __name__ == "__main__":
    data, labels = load_dataset(DATA_FOLDER, SAMPLING_RATE)
    clean_data, clean_labels = remove_bad_labels(data, labels)
    _ , selected_indices = select_best_lead(clean_data, SAMPLING_RATE)
    lead_names = [
        'I', 'II', 'III', 'AVR', 'AVL', 'AVF',
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
    ]

#plot simple histoogram of selected indices
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(selected_indices, bins=np.arange(0, 13)-0.5,
                color='skyblue', edgecolor='black')
    plt.xticks(range(12))
    plt.xlabel('Selected Lead Index')
    plt.ylabel('Number of Records')
    plt.xticks(ticks=range(12), labels=lead_names)
    plt.title('Histogram of Selected Leads with Fewest Peaks')
    plt.grid(axis='y')
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'selected_leads_histogram.png'))
    plt.close()
