# Downstream Classification Evaluation

This module evaluates the impact of ECG denoising on downstream classification performance.

## Overview

The evaluation pipeline:
1. Loads pre-trained **12-lead classification models** (xresnet1d101, inception1d)
2. Takes **validation fold** ECG data
3. Adds realistic noise to all 12 leads
4. Applies **denoising models** (trained on single leads) to each lead independently
5. Classifies the denoised 12-lead signals
6. Computes **macro-AUC** with **95% confidence intervals** using bootstrap

## Key Features

### Multi-Lead Processing
- Denoising models are trained on **single leads**
- Classification models use **all 12 leads**
- Each lead is denoised independently, then reconstructed into 12-lead format

### Sampling Rate Handling
- **Automatic resampling** if denoising and classification models use different sampling rates
- Warning messages inform you of any resampling operations
- Example: Denoising at 500Hz → Classification at 100Hz

### Normalization
- Uses the **same normalization** (StandardScaler) for both tasks
- No re-normalization needed between denoising and classification
- Preprocessing follows the base experiment's standardization

### Noise Configuration
- Uses the noise config specified in `config.yaml`
- Default: `../noise/config/default.yaml`
- Noise is added with `mode='eval'` to avoid data leakage

## Usage

### Basic Usage

```bash
cd code/denoising
python evaluate_downstream.py
```

This will:
- Use `config.yaml` for denoising configuration
- Look for classification models in `../../output/exp0/`
- Assume classification models trained at 100Hz

### Advanced Usage

```bash
# Specify different base experiment
python evaluate_downstream.py --base-exp exp1

# Specify different classification sampling rate
python evaluate_downstream.py --classification-fs 500

# Use custom config
python evaluate_downstream.py --config my_config.yaml --base-exp exp0 --classification-fs 100
```

### Parameters

- `--config`: Path to denoising experiment config file (default: `config.yaml`)
- `--base-exp`: Name of base classification experiment folder (default: `exp0`)
- `--classification-fs`: Sampling frequency of classification models in Hz (default: `100`)

## Requirements

### Pre-trained Models

You need:

1. **Denoising models**: Trained via `run_denoising_experiment.py`
   - Located in `output/exp_denoising/models/`
   - Models specified in `config.yaml`

2. **Classification models**: Pre-trained on PTB-XL
   - `fastai_xresnet1d101`
   - `fastai_inception1d`
   - Located in `../../output/exp0/models/` (or your specified base experiment)

### Data

- PTB-XL dataset in `../../data/ptbxl/`
- Noise data in `../../ecg_noise/data/`

## Output

Results are saved to `output/exp_denoising/downstream_results/`:

### 1. CSV Results
`downstream_classification_results.csv`:
```
denoising_model,classification_model,auc,auc_mean,auc_lower,auc_upper
clean,fastai_xresnet1d101,0.9234,0.9235,0.9156,0.9312
noisy,fastai_xresnet1d101,0.8891,0.8893,0.8802,0.8981
fcn,fastai_xresnet1d101,0.9012,0.9014,0.8925,0.9101
...
```

Columns:
- `denoising_model`: Name of denoising model (or 'clean'/'noisy' baseline)
- `classification_model`: Name of classification model
- `auc`: Point estimate of macro-AUC
- `auc_mean`: Bootstrap mean AUC
- `auc_lower`: Lower bound of 95% CI
- `auc_upper`: Upper bound of 95% CI

### 2. Comparison Plot
`downstream_classification_comparison.png`:
- Side-by-side horizontal bar plots
- One panel per classification model
- Error bars show 95% confidence intervals
- Colors: Green (clean), Red (noisy), Blue (Stage1), Purple (Stage2)

### 3. Improvement Heatmap
`downstream_improvement_heatmap.png`:
- Shows AUC improvement over noisy baseline
- Rows: Denoising models
- Columns: Classification models
- Color scale: Green (positive), Red (negative)

## Interpretation

### Baselines

1. **Clean baseline**: Upper bound performance (no noise)
2. **Noisy baseline**: Lower bound performance (noise, no denoising)

### Success Metrics

Good denoising should:
- **Recover performance**: AUC closer to clean than noisy
- **Statistical significance**: 95% CI doesn't overlap with noisy baseline
- **Consistency**: Works well across both classifiers

### Example Results

```
Model            | xresnet AUC | inception AUC | Avg Improvement
-----------------|-------------|---------------|----------------
Clean            | 0.9234      | 0.9187        | Baseline
Noisy            | 0.8891      | 0.8843        | -0.0354
fcn              | 0.9012      | 0.8967        | +0.0125
unet             | 0.9089      | 0.9034        | +0.0201
imunet           | 0.9123      | 0.9078        | +0.0247
drnet_fcn        | 0.9145      | 0.9101        | +0.0270
drnet_unet       | 0.9167      | 0.9124        | +0.0291
drnet_imunet     | 0.9189      | 0.9145        | +0.0312
```

In this example:
- Stage2 models consistently outperform Stage1
- DRnet_imunet recovers ~88% of lost performance (0.0312 / 0.0354)

## Technical Details

### Processing Pipeline

1. **Load 12-lead data** at classification sampling rate
2. **Apply StandardScaler** from base experiment
3. **Add noise** to all 12 leads using NoiseFactory
4. **Denoise each lead**:
   - Process leads independently
   - Use batch processing for efficiency
   - Shape: (n_samples, time, 1) → model → (n_samples, time, 1)
5. **Reconstruct 12-lead signal**
6. **Classify** using pre-trained classifier
7. **Compute metrics** with bootstrap confidence intervals

### Memory Management

- Denoising is done in batches (default: 32 samples)
- Only one denoised version kept in memory at a time
- Models are kept on GPU if available

### Bootstrap Confidence Intervals

- 100 bootstrap samples (default)
- Stratified sampling ensures all classes represented
- 95% confidence level
- Handles class imbalance gracefully

## Troubleshooting

### Issue: "Classification model not found"

**Solution**: Ensure you have run the base classification experiment:
```bash
cd code
python reproduce_results.py exp0
```

### Issue: "Denoising model not found"

**Solution**: Train denoising models first:
```bash
cd code/denoising
python run_denoising_experiment.py
```

### Issue: Sampling rate warnings

**Expected behavior**: Script automatically resamples if needed. This is normal if:
- Denoising trained at 500Hz
- Classification trained at 100Hz

### Issue: Out of memory

**Solutions**:
1. Reduce batch size in the script (line ~133): `batch_size=16`
2. Process fewer denoising models at once
3. Use CPU instead of GPU (in config.yaml: `use_cuda: false`)

### Issue: Low AUC scores

**Check**:
1. Are you using the correct base experiment?
2. Are labels loaded correctly?
3. Is normalization applied properly?
4. Compare with baseline (clean/noisy) first

## Integration with Paper

This evaluation answers the question:

> **"Does denoising ECG signals improve downstream diagnostic performance?"**

Key insights:
- Quantifies the **practical value** of denoising
- Tests **generalization** to real diagnostic tasks
- Compares **Stage1 vs Stage2** architectures on utility
- Demonstrates **robustness** across different classifiers

## References

- **PTB-XL**: Physikalisch-Technische Bundesanstalt database
- **Classification models**: xresnet1d101, inception1d architectures
- **Denoising models**: FCN, UNet, IMUnet, DRnet (Stage2)
- **Noise**: Real ECG noise from MIT-BIH, NSTDB

## Contact

For issues or questions about this evaluation module, please refer to the main repository documentation.
