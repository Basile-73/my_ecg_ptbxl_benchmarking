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
- **MECG-E (Phase)**: Mamba-based ECG Enhancer using magnitude+phase features (~2.8M params) - Stage1
- **MECG-E (Complex)**: Mamba-based ECG Enhancer using complex features (~2.8M params) - Stage1
- **MECG-E (Wav)**: Mamba-based ECG Enhancer using waveform features (~2.7M params) - Stage1
- **DRnet (Stage2)**: Two-stage reconstruction model - Uses Stage1 output + noisy signal

**Note on MECG-E**: A state-of-the-art Mamba-based denoiser with fast inference and excellent performance under noisy conditions. Based on "MECG-E: Mamba-based ECG Enhancer for Baseline Wander Removal" (arXiv:2409.18828).

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

### MECG-E Models

MECG-E is a state-of-the-art Mamba-based ECG denoiser integrated into this pipeline. It offers three variants based on different feature representations:

#### Available MECG-E Variants

```yaml
models:
  - name: "mecge_phase"
    type: "mecge_phase"  # Magnitude+phase STFT features (best performance)
    epochs: 50
    lr: 0.001
    batch_size: 32

  - name: "mecge_complex"
    type: "mecge_complex"  # Complex STFT features
    epochs: 50
    lr: 0.001
    batch_size: 32

  - name: "mecge_wav"
    type: "mecge_wav"  # Waveform features (fastest, simpler loss)
    epochs: 50
    lr: 0.001
    batch_size: 32
```

**Variant Details:**
- **`mecge_phase`**: Uses magnitude+phase STFT features with combined loss (time+complex+consistency). Typically achieves best performance.
- **`mecge_complex`**: Uses complex STFT features with combined loss. Good balance of performance and complexity.
- **`mecge_wav`**: Uses waveform features with time-domain loss only. Fastest training, simpler architecture.

#### Unique Training Interface

MECG-E has a different training interface compared to other models:
- **During training:** `forward(clean, noisy)` returns loss directly (no external MSE loss needed)
- **During inference:** `denoising(noisy)` returns denoised predictions

The pipeline automatically detects MECG-E models using `hasattr(model, 'denoising')` and handles them appropriately. **Users don't need to do anything special** - just specify the model type.

#### Configuration Requirements

MECG-E uses internal YAML configuration files located in `denoising_models/my_MECG-E/config/`:
- `MECGE_phase.yaml` - Configuration for phase variant
- `MECGE_complex.yaml` - Configuration for complex variant
- `MECGE_wav.yaml` - Configuration for waveform variant

These configs are automatically loaded based on the model type. Users can modify these files to adjust MECG-E hyperparameters such as:
- STFT parameters (n_fft, hop_length, win_length)
- Mamba block configuration (d_model, d_state, d_conv)
- Loss function weights (time_weight, complex_weight, consistency_weight)

#### Using MECG-E with Stage2 Models

MECG-E can serve as a Stage1 model for Stage2/DRnet refinement:

```yaml
models:
  # Stage 1: Train MECG-E
  - name: "mecge_phase"
    type: "mecge_phase"
    epochs: 50
    lr: 0.001
    batch_size: 32

  # Stage 2: Train DRnet using MECG-E output
  - name: "drnet_mecge"
    type: "stage2"
    stage1_model: "mecge_phase"  # Use MECG-E as Stage1
    epochs: 50
    lr: 0.001
    batch_size: 32
```

This combines MECG-E's strong denoising capabilities with Stage2 refinement for even better results.

#### Special Considerations

**Input/Output Format:**
- MECG-E accepts 4D tensors `(batch, 1, 1, time)` matching other Stage1 models
- Also supports 3D `(batch, 1, time)` for backward compatibility
- Output shape always matches input shape

**Loss Computation:**
- MECG-E computes loss internally using a combination of:
  - Time-domain loss: MSE between clean and denoised waveforms
  - Frequency-domain loss: MSE in STFT complex domain (phase/complex variants)
  - Consistency loss: Ensures STFT magnitude/phase consistency (phase/complex variants)
- Loss weights configured in YAML files

**Performance:**
- Inference speed: Faster than diffusion-based denoisers, comparable to CNN-based models
- Memory usage: STFT transformations may require more memory than simple CNNs
- Variable-length inputs: MECG-E handles different signal lengths natively through STFT (no wrapper needed)

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

### Train MECG-E Models

**Train single MECG-E variant:**
```yaml
models:
  - name: "mecge_phase"
    type: "mecge_phase"
    epochs: 50
    lr: 0.001
    batch_size: 32
```

**Compare all MECG-E variants:**
```yaml
models:
  - name: "mecge_phase"
    type: "mecge_phase"
    epochs: 50
    lr: 0.001
    batch_size: 32
  - name: "mecge_complex"
    type: "mecge_complex"
    epochs: 50
    lr: 0.001
    batch_size: 32
  - name: "mecge_wav"
    type: "mecge_wav"
    epochs: 50
    lr: 0.001
    batch_size: 32
```

**Compare MECG-E with traditional models:**
```yaml
models:
  - name: "fcn"
    type: "fcn"
    epochs: 50
    lr: 0.001
    batch_size: 32
  - name: "mecge_phase"
    type: "mecge_phase"
    epochs: 50
    lr: 0.001
    batch_size: 32
```

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

**MECG-E specific dependencies:**
- Mamba SSM package (installed from `denoising_models/my_MECG-E/mamba/`)
- CUDA >= 12.0 (recommended for optimal Mamba performance)
- Note: The Mamba package is included as part of the my_MECG-E submodule

Install from existing environment:
```bash
conda env create -f ../../ecg_env.yml
conda activate ecg_benchmarking

# For MECG-E support, additionally install Mamba:
cd denoising_models/my_MECG-E/mamba
pip install .
cd ../../..
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
- Hung et al. (2024): MECG-E: Mamba-based ECG Enhancer for Baseline Wander Removal (arXiv:2409.18828)

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

### MECG-E import errors
- Ensure the my_MECG-E submodule is properly initialized:
  ```bash
  git submodule update --init --recursive
  ```
- Check that Mamba package is installed:
  ```bash
  cd denoising_models/my_MECG-E/mamba && pip install .
  ```
- Verify CUDA version >= 12.0 for optimal Mamba support:
  ```bash
  nvcc --version
  ```

### MECG-E YAML config not found
- Config files should be in `denoising_models/my_MECG-E/config/`
- Check that the submodule includes: `MECGE_phase.yaml`, `MECGE_complex.yaml`, `MECGE_wav.yaml`
- If missing, reinitialize the submodule or check the repository structure

## Notes

- This pipeline uses **online noise generation** for training (new noise added each epoch)
- This provides better generalization since we have more training samples than noise samples
- NoiseFactory specifications are loaded from external config file (e.g., `default.yaml`)
- **Prevents data leakage**: Training uses 'train' mode noise, evaluation uses 'eval' mode noise
- Structure follows scp_experiment.py and noise_experiment.py patterns
- Results are reproducible with fixed random seeds
- GPU is automatically used if available
