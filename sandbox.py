from mycode.classification.utils.utils import load_dataset

data, labels = load_dataset(path='data/physionet.org/files/ptb-xl/1.0.3/', sampling_rate=500)
data.shape

from mycode.denoising.denoising_utils import bandpass_filter

data_bandpassed = bandpass_filter(data, fs=500)

# In the same sub plot: plot original and bandpassed signals for the first channel of the first sample. data has shape (num_samples, num_timesteps, num_channels)
from matplotlib import pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(data[0, :, 0], label='Original Signal', alpha=0.5)
plt.plot(data_bandpassed[0, :, 0], label='Bandpassed Signal', alpha=0.8)
plt.title('Original vs Bandpassed Signal (First Channel of First Sample)')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

data_val = data[labels.strat_fold == 9]
data_train = data[labels.strat_fold.isin(range(9))]
