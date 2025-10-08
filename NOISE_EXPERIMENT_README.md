# ECG Noise Robustness Experiment

## Overview

This framework tests the robustness of pre-trained ECG classification models to realistic physiological noise. It evaluates how well models maintain their performance when exposed to baseline wander, muscle artifacts, electrode motion, and additive white Gaussian noise.

**Key Features:**
- ✅ No model re-training required
- ✅ Realistic noise from MIT-BIH Noise Stress Test Database
- ✅ Bootstrap confidence intervals (90% CI with 100 samples)
- ✅ Supports all model types (fastai, wavelet)
- ✅ Parallel evaluation for speed

## Quick Start

```bash
cd code
python run_noise_experiments.py
```

This will test `fastai_xresnet1d101` with default noise settings and save results to `output/exp0_noise/`.

## Usage

### Simple Test (Single Model)
```bash
cd code
python run_noise_experiments.py --model fastai_xresnet1d101
```

### Test Multiple Models
```bash
# Test specific models
python run_noise_experiments.py --models fastai_xresnet1d101 fastai_resnet1d_wang

# Test all available models
python run_noise_experiments.py --all
```

### Quick Test (Fewer Bootstrap Samples)
```bash
python run_noise_experiments.py --quick
# Uses 20 bootstrap samples instead of 100 (~5 min vs ~20 min)
```

### Different Noise Configurations
```bash
# Default (realistic clinical noise)
python run_noise_experiments.py --noise-config default

# Light (less severe noise)
python run_noise_experiments.py --noise-config light
```

### Different Base Experiment
```bash
# Test models from exp1 (diagnostic task)
python run_noise_experiments.py --base-exp exp1 --all
```

### All Options
```bash
python run_noise_experiments.py \
  --base-exp exp0 \
  --models fastai_xresnet1d101 fastai_resnet1d_wang \
  --noise-config default \
  --n-bootstrap 100 \
  --n-jobs 20 \
  --sampling-rate 100
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--base-exp` | Base experiment name | `exp0` |
| `--model` | Single model to test | `fastai_xresnet1d101` |
| `--models` | Multiple models to test | - |
| `--all` | Test all available models | `False` |
| `--noise-config` | Noise configuration (default/light) | `default` |
| `--quick` | Quick test (20 bootstrap samples) | `False` |
| `--n-bootstrap` | Number of bootstrap samples | `100` |
| `--n-jobs` | Parallel jobs | `20` |
| `--sampling-rate` | Sampling frequency (100 or 500) | `100` |

## Python API

```python
from experiments.noise_experiment import NoiseRobustnessExperiment

# Create experiment
experiment = NoiseRobustnessExperiment(
    experiment_name='exp0_noise',
    base_experiment='exp0',
    model_names=['fastai_xresnet1d101'],
    datafolder='../data/ptbxl/',
    outputfolder='../output/',
    sampling_frequency=100
)

# Run pipeline
experiment.prepare()   # Load data and generate noisy versions
experiment.perform()   # Generate predictions on noisy data
summary = experiment.evaluate()  # Compute metrics with bootstrap CIs
```

## Understanding Results

### Summary File
Located at: `output/exp0_noise/results/noise_robustness_summary.csv`

```csv
model,split,clean_auc,noisy_auc,auc_drop,clean_auc_ci_lower,clean_auc_ci_upper,noisy_auc_ci_lower,noisy_auc_ci_upper
fastai_xresnet1d101,test,0.925,0.887,0.038,0.918,0.930,0.879,0.895
```

**Key Metrics:**
- `clean_auc`: Model performance on clean ECG data (baseline)
- `noisy_auc`: Model performance on noisy ECG data
- `auc_drop`: Performance degradation (clean_auc - noisy_auc)
- `*_ci_lower/upper`: 90% confidence interval bounds

**Interpretation:**
- **Small drop (0.02-0.04)**: Good noise robustness ✓
- **Medium drop (0.04-0.06)**: Average robustness
- **Large drop (>0.06)**: Poor noise robustness ✗

### Detailed Results
Each model has 4 detailed result files in `output/exp0_noise/models/{model_name}/results/`:
- `val_clean_results.csv` - Validation set, clean data
- `val_noisy_results.csv` - Validation set, noisy data
- `test_clean_results.csv` - Test set, clean data
- `test_noisy_results.csv` - Test set, noisy data

Each contains:
```csv
,macro_auc
point,0.925      # Performance on full dataset
mean,0.924       # Average across bootstrap samples
lower,0.918      # 5th percentile (90% CI lower bound)
upper,0.930      # 95th percentile (90% CI upper bound)
```

## Output Structure

```
output/exp0_noise/
├── data/
│   ├── X_val_clean.npy          # Original validation data
│   ├── X_val_noisy.npy          # Noisy validation data
│   ├── X_test_clean.npy         # Original test data
│   ├── X_test_noisy.npy         # Noisy test data
│   ├── y_val.npy                # Validation labels
│   └── y_test.npy               # Test labels
├── models/
│   └── {model_name}/
│       ├── y_val_clean_pred.npy    # Predictions on clean val
│       ├── y_val_noisy_pred.npy    # Predictions on noisy val
│       ├── y_test_clean_pred.npy   # Predictions on clean test
│       ├── y_test_noisy_pred.npy   # Predictions on noisy test
│       └── results/
│           ├── val_clean_results.csv
│           ├── val_noisy_results.csv
│           ├── test_clean_results.csv
│           └── test_noisy_results.csv
└── results/
    └── noise_robustness_summary.csv
```

## Noise Configuration

### Default Configuration
File: `ecg_noise/configs/default.yaml`

```yaml
SNR:  # Signal-to-Noise Ratio in dB
  bw: 2.5      # Baseline wander (0-5 dB range, realistic clinical)
  ma: 7.5      # Muscle artifacts (5-10 dB range)
  em: 12.5     # Electrode motion (10-15 dB range)
  AWGN: 22.5   # White Gaussian noise (20-25 dB range)
```

### Light Configuration
File: `ecg_noise/configs/light.yaml`

```yaml
SNR:
  bw: 5.0      # Less severe baseline wander
  ma: 10.0     # Less severe muscle artifacts
  em: 15.0     # Less severe electrode motion
  AWGN: 25.0   # Less severe white noise
```

Higher SNR = Less noise = Easier for models

## How It Works

### 1. Data Preparation
- Loads clean ECG data from base experiment
- Applies same preprocessing (standard scaling)
- Generates noisy versions using NoiseFactory
- Saves both clean and noisy data

### 2. Noise Addition
Four types of realistic noise are added simultaneously:
- **Baseline Wander (BW)**: Low-frequency drift
- **Muscle Artifacts (MA)**: High-frequency muscle activity
- **Electrode Motion (EM)**: Transient noise from electrode movement
- **AWGN**: Additive White Gaussian Noise

Each noise type is scaled to achieve target SNR:
```
x_noisy = x_clean + noise_BW + noise_MA + noise_EM + noise_AWGN
```

### 3. Model Evaluation
- Loads pre-trained models (no training!)
- Generates predictions on both clean and noisy data
- Evaluates performance using bootstrap confidence intervals
- Compares clean vs. noisy performance

### 4. Bootstrap Confidence Intervals
- Generates 100 bootstrap samples (default)
- Computes metrics for each sample
- Reports point estimate, mean, and 90% CI (5th-95th percentile)
- Ensures statistical rigor

## Performance Expectations

| Configuration | Bootstrap Samples | Time (1 model) | Time (All models) |
|--------------|-------------------|----------------|-------------------|
| Quick        | 20                | ~5 min         | ~30 min           |
| Standard     | 100               | ~20 min        | ~2 hours          |
| High Precision | 1000            | ~3 hours       | ~20 hours         |

## Prerequisites

Before running experiments:

1. **Base experiment completed**
   ```bash
   ls output/exp0/models/  # Should show model directories
   ls output/exp0/data/    # Should contain y_val.npy, standard_scaler.pkl
   ```

2. **PTB-XL data downloaded**
   ```bash
   ls data/ptbxl/raw100.npy  # Should exist
   ```

3. **Python environment**
   - PyTorch
   - NumPy, Pandas, SciPy
   - scikit-learn
   - fastai (for fastai models)
   - wfdb (for noise data download)

## Troubleshooting

### Error: "Model not found"
**Solution:** Ensure base experiment has been run first.
```bash
cd code
python reproduce_results.py
```

### Error: "No noise data"
**Solution:** First run auto-downloads from PhysioNet (requires internet).
Data is saved to `ecg_noise/data/` for future use.

### Out of Memory
**Solution:** Reduce parallel jobs or use quick mode.
```bash
python run_noise_experiments.py --quick --n-jobs 10
```

### Too Slow
**Solution:** Use quick mode or reduce bootstrap samples.
```bash
python run_noise_experiments.py --n-bootstrap 20
```

## Adding New Models

To test new model types, edit `noise_experiment.py`:

```python
def _predict_with_model(self, modelname, X_data):
    # ... existing code ...
    elif modelname.startswith('my_model_'):
        from models.my_model import MyModel
        model = MyModel(...)
        return model.predict(X_data)
```

## Examples

### Example 1: Compare All Models
```bash
python run_noise_experiments.py --all
cat output/exp0_noise/results/noise_robustness_summary.csv | column -t -s,
```

### Example 2: Quick Development Test
```bash
python run_noise_experiments.py --quick --model fastai_xresnet1d101
```

### Example 3: Compare Noise Levels
```bash
# Test with default noise
python run_noise_experiments.py --noise-config default

# Test with light noise
python run_noise_experiments.py --noise-config light

# Compare results manually
```

### Example 4: Test Different Experiments
```bash
# All task (exp0)
python run_noise_experiments.py --base-exp exp0 --all

# Diagnostic task (exp1)
python run_noise_experiments.py --base-exp exp1 --all

# Compare robustness across tasks
```

## Implementation Details

### NoiseRobustnessExperiment Class

**Methods:**
- `prepare()`: Load data, generate noisy versions
- `perform()`: Generate predictions with pre-trained models
- `evaluate()`: Compute metrics with bootstrap CIs

**Key Features:**
- Reuses existing utilities from `utils/utils.py`
- Loads models using existing model classes
- Applies same preprocessing as base experiment
- Uses NoiseFactory from `ecg_noise` package

### Code Structure

```
code/
├── experiments/
│   └── noise_experiment.py       # Main experiment class
└── run_noise_experiments.py      # CLI interface
```

## Citation

If you use this code, please cite:

- **PTB-XL Database**: Wagner et al. (2020)
- **MIT-BIH Noise Stress Test**: Moody et al. (1984)
- **Base Framework**: Strodthoff et al. (2021)

## License

Same as base PTB-XL benchmarking framework.

---

**For More Information:**
- See `code/experiments/noise_experiment.py` for implementation details
- Check `ecg_noise/` for noise generation code
- Review examples in `code/run_noise_experiments.py`
