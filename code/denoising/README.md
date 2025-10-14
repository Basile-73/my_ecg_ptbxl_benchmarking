# ECG Denoising Pipeline

A simplified PyTorch-based pipeline for training and evaluating ECG denoising models on PTB-XL dataset.

## Overview

This pipeline implements the denoising approach described in the selection_denoising notebook:
- Preprocesses PTB-XL ECG signals (bad label removal, lead selection, bandpass filtering)
- Uses **online noise generation** (adds new noise each epoch for better generalization)
- Trains deep learning models (FCN, UNet, IMUnet, Stage2)
- Evaluates using SNR improvement and RMSE metrics
- Noise specifications loaded from external config file (e.g., `code/noise/config/default.yaml`)

## File Structure

```
code/denoising/
├── config.yaml                      # Configuration file
├── run_denoising_experiment.py      # Main experiment runner
├── evaluate_results.py              # Evaluation and plotting
├── README.md                        # This file
├── denoising_utils/
│   ├── preprocessing.py             # ECG preprocessing utilities
│   ├── training.py                  # Model training functions
│   └── utils.py                     # Metrics, Dataset, model loading
└── ECG-processing/                  # Model implementations
```

## Quick Start

### 1. Configure Experiment

Edit `config.yaml` to set:
- `experiment_name`: Name for your experiment
- `noise_config_path`: Path to noise config file (e.g., `../noise/config/default.yaml`)
- `models`: Which models to train (FCN, UNet, IMUnet, Stage2)

### 2. Run Experiment

```bash
python run_denoising_experiment.py
```

This will:
1. Load and preprocess PTB-XL data
2. Initialize NoiseFactory with noise config file
3. Train all specified models with **online noise generation** (new noise each epoch)
4. Save trained models and predictions

### 3. Evaluate Results

```bash
python evaluate_results.py
```

For PDF report:
```bash
python evaluate_results.py --report
```

Results saved to `output/{experiment_name}/results/`:
- `summary.csv`: Mean metrics per model
- `detailed_results.csv`: Per-sample results
- `comparison_plots.png`: Visual comparison
- `example_denoising.png`: Example signals
- `denoising_report.pdf`: Full report (if --report used)

## Configuration

### Key Settings

```yaml
experiment_name: "denoising_exp"
sampling_frequency: 500
random_seed: 42

# Path to noise configuration
noise_config_path: "../noise/config/default.yaml"

noise:
  types: ['bw', 'em', 'ma']  # Noise types to use
  snr_levels: [-5, 0, 5, 10]  # SNR levels (defined in noise config)
  mode: "test"                # Which noise data split to use

preprocessing:
  bad_label_removal: true
  lead_selection: true    # Select lead with fewest peaks
  bandpass_lowcut: 1.0
  bandpass_highcut: 45.0
  bandpass_order: 2

training:
  num_epochs: 50
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 10
```

### Models

Each model has its own configuration:

```yaml
models:
  - name: "fcn"
    type: "fcn"
    epochs: 50
    lr: 0.001
    batch_size: 32
```

Available models:
- **FCN**: Fast Convolutional Network (~500K params) - Stage1
- **UNet**: U-Net architecture (~600K params) - Stage1
- **IMUnet**: Improved U-Net (~1M params) - Stage1
- **DRnet (Stage2)**: Two-stage reconstruction model - Uses Stage1 output + noisy signal

### Two-Stage Training

Stage2 (DRnet) requires a trained Stage1 model. Specify which Stage1 model's predictions to use:

```yaml
models:
  # Stage 1: Train base denoising model
  - name: "fcn"
    type: "fcn"
    epochs: 50
    lr: 0.001
    batch_size: 32

  # Stage 2: Train DRnet using FCN's predictions
  - name: "drnet_fcn"
    type: "stage2"           # or "drnet"
    stage1_model: "fcn"      # Use FCN's output as input
    epochs: 50
    lr: 0.001
    batch_size: 32
```

**Stage2 Training Pipeline:**
1. Train Stage1 model (e.g., FCN) with noisy input → clean output
2. Freeze Stage1 weights
3. Train Stage2 (DRnet) with 2-channel input:
   - Channel 1: Noisy signal
   - Channel 2: Stage1 (FCN) prediction
4. DRnet output → final denoised signal

You can train multiple Stage2 models using different Stage1 models:
```yaml
models:
  - name: "fcn"
    type: "fcn"
    ...
  - name: "unet"
    type: "unet"
    ...
  - name: "drnet_fcn"
    type: "stage2"
    stage1_model: "fcn"    # DRnet using FCN
    ...
  - name: "drnet_unet"
    type: "stage2"
    stage1_model: "unet"   # DRnet using UNet
    ...
```

**Training Stage2 Separately (Avoiding Compute Waste):**

Stage2 models can use Stage1 models from previous runs without retraining them:

1. **Automatic Detection** - If Stage1 model exists in output folder:
   ```yaml
   # Only train Stage2 - will find pre-trained fcn automatically
   models:
     - name: "drnet_fcn"
       type: "stage2"
       stage1_model: "fcn"  # Looks in output/exp_denoising/models/fcn/best_model.pth
   ```

2. **Custom Path** - Specify exact path to pre-trained Stage1:
   ```yaml
   models:
     - name: "drnet_fcn"
       type: "stage2"
       stage1_model: "fcn"
       stage1_model_path: "path/to/pretrained/fcn/model.pth"  # Custom location
   ```

3. **Training Both** - Stage1 and Stage2 in same run:
   ```yaml
   models:
     - name: "fcn"
       type: "fcn"
       ...
     - name: "drnet_fcn"
       type: "stage2"
       stage1_model: "fcn"  # Uses fcn from current run
   ```

This allows flexible experimentation without wasting compute on retraining Stage1 models.

### Noise Configuration

Noise specifications are defined in a separate YAML file (e.g., `code/noise/config/default.yaml`):

```yaml
SNR:  # Target SNR levels for each noise type
  bw: 2.5    # Baseline wander (0-5 dB range)
  ma: 7.5    # Muscle artifact (5-10 dB range)
  em: 12.5   # Electrode motion (10-15 dB range)
  AWGN: 22.5 # White noise (20-25 dB range)
```

The main config file points to this noise config via `noise_config_path`.

## Preprocessing Details

Following Hu et al. (2024) and Dias et al. (2024):

1. **Bad Label Removal**: Remove signals with bad quality labels
2. **Lead Selection**: Select single lead with fewest detected peaks
3. **Bandpass Filter**: Butterworth filter (1-45 Hz, order 2)
4. **Normalization**: Z-score normalization per signal

## Online Noise Generation

Unlike traditional approaches that pre-generate a fixed noisy dataset, this pipeline uses **online noise generation**:

- **During Training**: New noise is generated for each batch in every epoch using `mode='train'`
- **Benefits**:
  - Better generalization (model sees many noise realizations)
  - More effective use of limited noise samples
  - Reduces overfitting to specific noise patterns
- **During Evaluation**: Uses `mode='eval'` to generate test noise (prevents data leakage)

This is particularly important because we have many more training ECG samples than noise samples available.

### Preventing Data Leakage

The pipeline carefully manages noise samples to avoid data leakage:

1. **Training**: Uses `mode='train'` noise samples
2. **Validation**: Uses `mode='train'` noise samples (online generation)
3. **Evaluation**: Uses `mode='eval'` noise samples (completely separate from training)

This ensures that the evaluation noise has never been seen during training, providing an unbiased assessment of model performance.## Metrics

- **SNR Improvement (dB)**: Output SNR - Input SNR
- **RMSE**: Root Mean Square Error between clean and denoised
- **RMSE Improvement (%)**: Percentage reduction in RMSE

## Output Structure

```
output/{experiment_name}/
├── data/
│   ├── clean_train.npy
│   ├── clean_test.npy
│   ├── noisy_train.npy
│   └── noisy_test.npy
├── models/
│   ├── fcn/
│   │   ├── best_model.pth
│   │   ├── predictions.npy
│   │   └── training_log.csv
│   └── unet/
│       └── ...
└── results/
    ├── summary.csv
    ├── detailed_results.csv
    ├── comparison_plots.png
    ├── example_denoising.png
    └── denoising_report.pdf
```

## Usage Examples

### Use different noise configurations

Create custom noise config file (e.g., `custom_noise.yaml`):
```yaml
SNR:
  bw: 0.0    # Lower SNR for harder task
  ma: 5.0
  em: 10.0
```

Then update `config.yaml`:
```yaml
noise_config_path: "../noise/config/custom_noise.yaml"
```

### Train specific models

Edit `config.yaml`:
```yaml
models:
  - name: "fcn"
    type: "fcn"
    epochs: 50
    lr: 0.001
    batch_size: 32
  # - name: "unet"
  #   type: "unet"
  #   ...
  # Comment out models you don't want to train
```

### Evaluate multiple experiments

```bash
# Edit config.yaml to set experiment_name
python evaluate_results.py --config config.yaml
```

## Dependencies

Core requirements:
- Python 3.8+
- PyTorch 1.10+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- PyYAML
- SciPy
- tqdm

Install from existing environment:
```bash
conda env create -f ../../ecg_env.yml
conda activate ecg_benchmarking
```

## Reproducibility

All experiments are seeded for reproducibility:
- Random seed set in config.yaml
- PyTorch deterministic mode enabled
- NumPy random seed set

## References

- PTB-XL: A large publicly available electrocardiography dataset
- Hu et al. (2024): Bad label removal for ECG signals
- Dias et al. (2024): Lead selection based on peak detection

## Troubleshooting

### CUDA out of memory
- Reduce `batch_size` in config.yaml
- Set `device: "cpu"` for CPU training

### Import errors from ECG-processing
- Models are loaded dynamically using sys.path manipulation
- Ensure ECG-processing folder is in code/denoising/

### NoiseFactory not found
- Ensure ecg_noise_factory is installed:
  ```bash
  pip install -e ../../ecg_noise
  ```

## Notes

- This pipeline uses **online noise generation** for training (new noise added each epoch)
- This provides better generalization since we have more training samples than noise samples
- NoiseFactory specifications are loaded from external config file (e.g., `default.yaml`)
- **Prevents data leakage**: Training uses 'train' mode noise, evaluation uses 'eval' mode noise
- Structure follows scp_experiment.py and noise_experiment.py patterns
- Results are reproducible with fixed random seeds
- GPU is automatically used if available
