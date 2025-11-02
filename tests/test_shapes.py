# load ../data/physionet.org/files/ptb-xl/1.0.3/raw100.npy and print its shape
import numpy as np
data = np.load('data/physionet.org/files/ptb-xl/1.0.3/raw100.npy', allow_pickle=True)
print(data.shape)

data = np.load('mycode/denoising/output/exp_denoising_norm_channel/data/clean_train.npy', allow_pickle=True)
print(data.shape)
