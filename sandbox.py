import numpy as np

y_val = np.load('mycode/classification/output/refactored500/exp1.1.1/data/y_val.npy', allow_pickle=True)

from mycode.classification.utils.utils import load_dataset, compute_label_aggregations, select_data
datafolder = 'data/physionet.org/files/ptb-xl/1.0.3/'
task = 'superdiagnostic'

data, raw_labels = load_dataset(datafolder, 100)
labels = compute_label_aggregations(raw_labels, datafolder, task)
data, labels, y, _ = select_data(data, labels, task, 0, 'noise/')
