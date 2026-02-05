import numpy as np
from sklearn.metrics import roc_auc_score

y_val = np.load('mycode/classification/output/refactored500/exp1.1.1/data/y_val.npy', allow_pickle=True)
y_pred = np.load('mycode/classification/output/refactored/exp1.1.1/models/fastai_inception1d/y_val_pred.npy', allow_pickle=True)

from mycode.classification.utils.utils import load_dataset, compute_label_aggregations, select_data
datafolder = 'data/physionet.org/files/ptb-xl/1.0.3/'
task = 'superdiagnostic'

data, raw_labels = load_dataset(datafolder, 100)
labels = compute_label_aggregations(raw_labels, datafolder, task)
data, labels, y, _ = select_data(data, labels, task, 0, 'noise/')

# get the mlb from mycode/classification/output/refactored/exp1.1.1/data/mlb.pkl
import pickle
with open('mycode/classification/output/refactored500/exp1.1.1/data/mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)


from mycode.denoising.denoising_utils.downstream import roc_by_class
from mycode.denoising.evaluate_downstream import compute_bootstrap_ci

# Option 1
test = roc_by_class(y_val, y_pred, mlb, classifyer_name='clean', densoising_model_name='fastai_inception1d')

test1 = test
test2 = test

test_list = test1 + test2
# concat in one pandas dataframe
import pandas as pd
df = pd.DataFrame(test_list)

# Option 2
# alternative to reuse more code
test2 = {}
for i, class_name in enumerate(mlb.classes_):
    y_val_class = y_val[:, i]
    y_pred_class = y_pred[:, i]
    res = roc_auc_score(y_val_class, y_pred_class, average='macro')
    test2[class_name] = res

# Option 1 and 2 produce the same results

# Bootstrap CIs
ci_results = {}
for i, class_name in enumerate(mlb.classes_):
    y_val_class = y_val[:, i]
    y_pred_class = y_pred[:, i]
    ci = compute_bootstrap_ci(y_val_class, y_pred_class, n_bootstraps=1000, confidence_level=0.95, metric='auc')
    ci_results[class_name] = ci
