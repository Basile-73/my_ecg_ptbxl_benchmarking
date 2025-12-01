I have created the following plan after thorough exploration and analysis of the codebase. Follow the below plan verbatim. Trust the files and references. Do not re-verify what's written in the plan. Explore only when absolutely necessary. First implement all the proposed file changes and then I'll review all the changes together at the end.



### Summary

# Per-Record Downstream Classification Evaluation

## Overview

The `evaluate_downstream_by_record.py` script extends the downstream classification evaluation by generating **per-record, per-diagnostic-class results** instead of aggregate metrics. This enables fine-grained analysis of which specific records and diagnostic classes benefit from denoising.

## Diagnostic-Type-Specific Experiments

**IMPORTANT**: This script now uses **different classification experiments for different diagnostic types** to ensure correct label encoding:

| Diagnostic Type | Classification Experiment | Classes Included |
|----------------|---------------------------|------------------|
| **Superdiagnostic** | `exp1.1.1` | NORM, MI, STTC, CD, HYP (5 classes) |
| **Subdiagnostic** | `exp1.1` | Subdiagnostic classes (23 classes) |
| **Diagnostic** | `exp1` | All diagnostic codes (44 classes) |

### Why This Is Necessary

Each classification experiment (`exp1.1.1`, `exp1.1`, `exp1`) is trained on a different subset of classes. The `MultiLabelBinarizer` (mlb.pkl) saved during training encodes the class-to-index mapping. When computing per-record correctness, we must use the **same mlb** that was used during training to ensure correct interpretation of prediction arrays.

**Previous behavior (INCORRECT)**: Used `exp0` for all diagnostic types, causing many classes to be skipped because they weren't in exp0's mlb.

**New behavior (CORRECT)**: Uses exp1.1.1/exp1.1/exp1 for super/sub/diagnostic types, ensuring all classes are properly evaluated.

### Prerequisites

Before running this script, you must train the classification experiments:

```bash
cd mycode/classification
python reproduce_results.py  # Trains exp0, exp1, exp1.1, exp1.1.1, exp2, exp3
```

This creates the required experiment folders with mlb.pkl, y_val.npy, and trained models.

## What This Script Does

Unlike `evaluate_downstream.py` which computes aggregate AUC metrics, this script:

1. **Loads pre-trained 12-lead classification models** (e.g., xresnet1d101, inception1d)
2. **Applies multiple noise configurations** to validation fold ECG data
3. **Denoises each lead independently** using trained denoising models
4. **Classifies signals** with clean, noisy, and denoised versions
5. **Computes per-record correctness** for each diagnostic class (binary: 1=correct, 0=wrong)
6. **Expands multi-label records** into multiple rows (one per diagnostic class)
7. **Saves detailed DataFrames** as CSV files for downstream analysis

## Key Differences from `evaluate_downstream.py`

| Feature | `evaluate_downstream.py` | `evaluate_downstream_by_record.py` |
|---------|-------------------------|-----------------------------------|
| **Output granularity** | Aggregate AUC per model | Per-record, per-class correctness |
| **Output format** | Single CSV with summary stats | 3 × n_noise_configs CSV files |
| **Multi-label handling** | Computes macro-AUC | Expands into separate rows |
| **Noise configs** | Single default config | Multiple configs from `qui_plot` |
| **Use case** | Overall model comparison | Record-level error analysis |

## Data Processing Pipeline

### Phase 1: Data Loading and Preparation

```
1. Load PTB-XL validation fold data
   ├─ Filter to validation fold (e.g., fold 9)
   ├─ Extract record IDs (ecg_id from index)
   └─ Apply label aggregations for 3 diagnostic types:
      ├─ superdiagnostic (e.g., NORM, MI, STTC)
      ├─ subdiagnostic (e.g., IMI, AMI, LVH)
      └─ diagnostic (all diagnostic codes)

2. Filter records with labels
   ├─ Keep only records with ≥1 label in any type
   └─ Align X_val, record_ids, and label_dfs

3. Load classification experiment data
   ├─ Load y_val (ground truth multi-hot labels)
   ├─ Load MultiLabelBinarizer (mlb.pkl) for class mapping
   └─ Apply StandardScaler from classification experiment
```

### Phase 2: Prediction Generation (Cached)

For **each noise configuration** in `qui_plot`:

```
1. Create NoiseFactory with noise config
   └─ Generate noisy validation data

2. Generate CLEAN baseline predictions
   ├─ For each classifier (e.g., xresnet1d101, inception1d):
   │  ├─ Load classification model
   │  ├─ Predict on clean data → y_pred_clean
   │  └─ Cache predictions
   └─ Clean up model from memory

3. Generate NOISY baseline predictions
   ├─ For each classifier:
   │  ├─ Load classification model
   │  ├─ Predict on noisy data → y_pred_noisy
   │  └─ Cache predictions
   └─ Clean up model from memory

4. Generate DENOISED predictions
   ├─ For each denoising model (e.g., fcn, unet, mecge_phase):
   │  ├─ Load denoising model (+ Stage1 if Stage2)
   │  ├─ Denoise noisy data (lead-by-lead)
   │  ├─ For each classifier:
   │  │  ├─ Load classification model
   │  │  ├─ Predict on denoised data → y_pred_denoised
   │  │  └─ Cache predictions
   │  └─ Clean up models from memory
   └─ Clear Stage1 cache
```

**Memory optimization**: Models are loaded one at a time, predictions are cached, then models are deleted with `torch.cuda.empty_cache()`.

### Phase 3: Per-Record Correctness Computation

For **each noise configuration**:

```
1. For each (denoising_model, classifier) combination:
   ├─ Retrieve cached predictions: y_pred (n_samples × n_classes)
   ├─ For each diagnostic type (super/sub/diagnostic):
   │  ├─ For each record:
   │  │  ├─ Get assigned diagnostic classes from label_df
   │  │  ├─ For each diagnostic class:
   │  │  │  ├─ Map class name → mlb index
   │  │  │  ├─ Get y_true[record_idx, class_idx]
   │  │  │  ├─ Get y_pred[record_idx, class_idx]
   │  │  │  ├─ Apply threshold (0.5): binary_pred = (y_pred ≥ 0.5)
   │  │  │  ├─ Compute correctness: (y_true == binary_pred) ? 1 : 0
   │  │  │  └─ Store: correctness_dict[(record_id, class)] = correctness
   │  │  └─ Skip classes not in mlb (with warning)
   │  └─ Return correctness_dict
   └─ Repeat for all diagnostic types
```

### Phase 4: DataFrame Expansion and Saving

For **each diagnostic type** (superdiagnostic, subdiagnostic, diagnostic):

```
1. Create base DataFrame
   ├─ For each record_id:
   │  ├─ Get list of diagnostic classes
   │  └─ Create one row per class:
   │     └─ {record_id: <id>, diagnostic_class: <class>}
   └─ Result: Expanded DataFrame (n_rows = total class assignments)

2. Add correctness columns
   ├─ For each (denoising_model, classifier) combination:
   │  ├─ Column name: "{model}_{classifier}"
   │  │  (e.g., "clean_fastai_xresnet1d101", "fcn_fastai_inception1d")
   │  └─ Populate with correctness values (0/1/NaN)
   └─ Result: DataFrame with record_id, diagnostic_class, and model columns

3. Save to CSV
   └─ Filename: "{diagnostic_type}_{noise_config_name}_by_record.csv"
      (e.g., "superdiagnostic_light_by_record.csv")
```

## Output Structure

### Files Generated

For a config with 3 noise configs (light, default, heavy):

```
output/{experiment_name}/downstream_results/by_sample/
├── superdiagnostic_light_by_record.csv
├── superdiagnostic_default_by_record.csv
├── superdiagnostic_heavy_by_record.csv
├── subdiagnostic_light_by_record.csv
├── subdiagnostic_default_by_record.csv
├── subdiagnostic_heavy_by_record.csv
├── diagnostic_light_by_record.csv
├── diagnostic_default_by_record.csv
└── diagnostic_heavy_by_record.csv
```

**Total files**: 3 diagnostic types × n_noise_configs

### CSV Format

Each CSV has the following structure:

```csv
record_id,diagnostic_class,clean_fastai_xresnet1d101,clean_fastai_inception1d,noisy_fastai_xresnet1d101,noisy_fastai_inception1d,fcn_fastai_xresnet1d101,fcn_fastai_inception1d,...
12345,NORM,1,1,0,0,1,1,...
12345,MI,1,0,0,0,0,1,...
12346,NORM,1,1,1,1,1,1,...
12347,STTC,0,1,0,0,0,0,...
12347,LVH,1,1,0,0,1,1,...
```

**Columns**:
- `record_id`: PTB-XL ecg_id
- `diagnostic_class`: Diagnostic class name (e.g., NORM, MI, STTC)
- `{model}_{classifier}`: Binary correctness (1=correct, 0=wrong, NaN=skipped)

**Rows**: One row per (record, diagnostic_class) pair

### Example: Multi-Label Expansion

If record `12345` has labels `[NORM, MI]`:

```
Original (1 record):
  record_id: 12345
  labels: [NORM, MI]

Expanded (2 rows):
  Row 1: record_id=12345, diagnostic_class=NORM, clean_xresnet=1, noisy_xresnet=0, ...
  Row 2: record_id=12345, diagnostic_class=MI, clean_xresnet=1, noisy_xresnet=0, ...
```

## Usage

### Basic Usage (Recommended)

```bash
cd mycode/denoising
python evaluate_downstream_by_record.py
```

**Default behavior** (uses diagnostic-type-specific experiments):
- Superdiagnostic classes: uses `exp1.1.1`
- Subdiagnostic classes: uses `exp1.1`
- Diagnostic classes: uses `exp1`
- Config: `code/denoising/configs/denoising_config.yaml`
- Classification sampling rate: `100` Hz
- Classifiers: `fastai_xresnet1d101`, `fastai_inception1d`

### Evaluate All Classifiers

```bash
python evaluate_downstream_by_record.py --classifiers all
```

Evaluates all 6 models:
- `fastai_xresnet1d101`
- `fastai_inception1d`
- `fastai_resnet1d_wang`
- `fastai_lstm`
- `fastai_lstm_bidir`
- `fastai_fcn_wang`

### Custom Configuration

```bash
# With custom config and all classifiers
python evaluate_downstream_by_record.py \
    --config configs/all_models_100hz_resampled_1.yaml \
    --classifiers all

# Use different sampling rate
python evaluate_downstream_by_record.py --classification-fs 500 --classifiers all

# Specific classifiers only
python evaluate_downstream_by_record.py --classifiers fastai_xresnet1d101 fastai_lstm
```

### Command-Line Arguments

- `--config`: Path to denoising config file (default: `code/denoising/configs/denoising_config.yaml`)
- `--base-exp`: **Legacy parameter**, only used if `use_diagnostic_specific_exps=False` in code (not recommended)
- `--classification-fs`: Classification sampling rate in Hz (default: `100`)
- `--classifiers`: Space-separated classifier names or `all` (default: `fastai_xresnet1d101 fastai_inception1d`)

## Configuration Requirements

### Denoising Config YAML

The script reads noise configurations from the `qui_plot` section:

```yaml
evaluation:
  qui_plot:
    enabled: true
    noise_configs:
      - name: 'light'
        path: 'noise/configs/light.yaml'
      - name: 'default'
        path: 'noise/configs/default.yaml'
      - name: 'heavy'
        path: 'noise/configs/heavy.yaml'
```

**Each noise config generates 3 DataFrames** (one per diagnostic type).

### Required Files

1. **Denoising models**: Trained models in `output/{experiment_name}/models/`
2. **Classification models**: Pre-trained in `output/{base_exp}/models/`
3. **Classification data**:
   - `output/{base_exp}/data/y_val.npy` (ground truth labels)
   - `output/{base_exp}/data/mlb.pkl` (MultiLabelBinarizer)
   - `output/{base_exp}/data/standard_scaler.pkl` (StandardScaler)
4. **PTB-XL data**: In `datafolder` specified in config
5. **Noise data**: In `noise_data_path` specified in config

## Technical Details

### Multi-Label Handling

PTB-XL records can have multiple diagnostic labels. The script handles this by:

1. **Expansion**: Each record with N labels becomes N rows
2. **Per-class evaluation**: Each row represents one (record, class) pair
3. **Independent correctness**: Each class is evaluated independently using threshold 0.5

**Example**:
- Record has labels: `[NORM, MI]`
- Prediction probabilities: `{NORM: 0.8, MI: 0.3, STTC: 0.1}`
- Binary predictions (threshold 0.5): `{NORM: 1, MI: 0, STTC: 0}`
- Correctness:
  - NORM: `1 == 1` → **1** (correct)
  - MI: `1 == 0` → **0** (wrong)

### Class Mapping with MultiLabelBinarizer

The script uses the **exact class ordering** from the classification experiment's MultiLabelBinarizer:

```python
# Load mlb from classification experiment
with open('output/exp0/data/mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)

# Map class names to indices
class_to_idx = {cls: idx for idx, cls in enumerate(mlb.classes_)}

# Get prediction for specific class
class_idx = class_to_idx['NORM']
pred_prob = y_pred[record_idx, class_idx]
```

**Why this matters**: Label aggregation may produce classes in different order than the MLB. Using MLB ensures correct alignment with `y_true` and `y_pred` arrays.

### Lead-by-Lead Denoising

Denoising models are trained on **single leads**, but classification uses **12 leads**:

```python
# Input: X_val_noisy (n_samples, n_timesteps, 12)
# Process each lead independently
for lead_idx in range(12):
    lead_data = X_val_noisy[:, :, lead_idx:lead_idx+1]  # (n_samples, n_timesteps, 1)

    # Denoise this lead
    denoised_lead = denoising_model(lead_data)

    # Store in output
    X_val_denoised[:, :, lead_idx] = denoised_lead

# Output: X_val_denoised (n_samples, n_timesteps, 12)
```

### Stage2 Model Handling

Stage2 models (e.g., DRnet) require **Stage1 model output**:

```python
# Load Stage1 model (cached for efficiency)
stage1_model = load_stage1_model(stage1_name)

# For each lead:
stage1_output = stage1_model(noisy_lead)  # (batch, 1, 1, time)

# Concatenate noisy + stage1 output
stage2_input = torch.cat([noisy_lead, stage1_output], dim=1)  # (batch, 2, 1, time)

# Pass through Stage2
denoised = stage2_model(stage2_input)  # (batch, 1, 1, time)
```

**Stage1 caching**: Stage1 models are loaded once per noise config and reused across all Stage2 models.

### Memory Management

The script uses a **two-phase approach** to minimize memory usage:

**Phase 1 (Generation)**:
- Load one model at a time
- Generate predictions
- Cache predictions as numpy arrays
- Delete model and clear GPU cache

**Phase 2 (Computation)**:
- Reuse cached predictions
- No model loading required
- Compute correctness from cached arrays

This allows processing many models without running out of memory.

## Use Cases

### 1. Error Analysis

Identify which records are consistently misclassified:

```python
import pandas as pd

# Load results
df = pd.read_csv('superdiagnostic_default_by_record.csv')

# Find records where all models fail
model_cols = [c for c in df.columns if c not in ['record_id', 'diagnostic_class']]
df['total_correct'] = df[model_cols].sum(axis=1)
df['total_models'] = df[model_cols].notna().sum(axis=1)

# Records with 0% accuracy across all models
hard_cases = df[df['total_correct'] == 0]
print(f"Hard cases: {len(hard_cases)} record-class pairs")
```

### 2. Model Comparison

Compare which models perform best on specific diagnostic classes:

```python
# Filter to specific class
norm_df = df[df['diagnostic_class'] == 'NORM']

# Compute accuracy per model
for col in model_cols:
    accuracy = norm_df[col].mean()
    print(f"{col}: {accuracy:.3f}")
```

### 3. Noise Sensitivity Analysis

Compare performance across noise levels:

```python
# Load all noise configs
light = pd.read_csv('superdiagnostic_light_by_record.csv')
default = pd.read_csv('superdiagnostic_default_by_record.csv')
heavy = pd.read_csv('superdiagnostic_heavy_by_record.csv')

# Compare clean baseline across noise levels
print(f"Clean accuracy (light): {light['clean_fastai_xresnet1d101'].mean():.3f}")
print(f"Clean accuracy (default): {default['clean_fastai_xresnet1d101'].mean():.3f}")
print(f"Clean accuracy (heavy): {heavy['clean_fastai_xresnet1d101'].mean():.3f}")
```

### 4. Denoising Benefit Analysis

Identify which records benefit most from denoising:

```python
# Compute improvement: denoised - noisy
df['improvement'] = df['fcn_fastai_xresnet1d101'] - df['noisy_fastai_xresnet1d101']

# Records that improved
improved = df[df['improvement'] > 0]
print(f"Improved: {len(improved)} / {len(df)} ({100*len(improved)/len(df):.1f}%)")

# Records that degraded
degraded = df[df['improvement'] < 0]
print(f"Degraded: {len(degraded)} / {len(df)} ({100*len(degraded)/len(df):.1f}%)")
```

## Troubleshooting

### Issue: "No validation samples found for fold X"

**Cause**: Invalid validation fold number in config.

**Solution**: Check `val_fold` in config.yaml. PTB-XL uses folds 1-10.

### Issue: "Skipped N diagnostic classes not in mlb"

**Cause**: Label aggregation produced classes not in classification training set.

**Expected behavior**: This is normal if classification was trained on a subset of classes. The script logs skipped classes and continues.

### Issue: Out of memory during prediction generation

**Solutions**:
1. Reduce number of classifiers: `--classifiers fastai_xresnet1d101`
2. Process fewer denoising models (comment out in config)
3. Reduce batch size in `denoise_12lead_signal()` (line 102, default=32)
4. Use CPU: Set `use_cuda: false` in config

### Issue: NaN values in output CSV

**Causes**:
1. Model failed to load (check logs for warnings)
2. Prediction generation failed (check error messages)
3. Diagnostic class not in MLB (expected, see "Skipped classes" warning)

**Check**: Look for `⚠️` warnings in console output.

### Issue: Mismatched sampling rates

**Expected behavior**: Script warns if denoising and classification use different rates. This is normal and handled automatically.

**Example**:
```
⚠️  WARNING: Sampling rate mismatch!
   Denoising models trained at: 500Hz
   Classification models trained at: 100Hz
```

The script loads data at classification sampling rate, so no resampling is needed.

## Performance Considerations

### Runtime

For a typical setup:
- 3 noise configs
- 5 denoising models
- 2 classifiers
- ~2000 validation samples

**Estimated time**: 30-60 minutes (with GPU)

**Breakdown**:
- Phase 1 (prediction generation): ~80% of time
- Phase 2 (correctness computation): ~20% of time

### Disk Space

Each CSV file size depends on:
- Number of records
- Number of diagnostic classes per record
- Number of model combinations

**Typical size**: 1-5 MB per CSV

**Total**: 3 × n_noise_configs × (1-5 MB)

### Memory Usage

**Peak memory** (during prediction generation):
- Clean/noisy data: ~500 MB (for 2000 samples at 100Hz)
- One classification model: ~100-500 MB
- Cached predictions: ~50 MB per (model, classifier) pair

**Total peak**: ~1-2 GB (with sequential model loading)

## Integration with Downstream Analysis

This script complements `evaluate_downstream.py`:

| Script | Output | Use Case |
|--------|--------|----------|
| `evaluate_downstream.py` | Aggregate AUC with CI | Overall model comparison |
| `evaluate_downstream_by_record.py` | Per-record correctness | Error analysis, record-level insights |

**Workflow**:
1. Run `evaluate_downstream.py` for high-level comparison
2. Run `evaluate_downstream_by_record.py` for detailed analysis
3. Use CSV outputs for custom visualizations and statistical tests

## References

- **PTB-XL Dataset**: Wagner et al. (2020) - Large publicly available ECG dataset
- **Label Aggregations**: `compute_label_aggregations()` from `utils/utils.py`
- **MultiLabelBinarizer**: Scikit-learn's multi-label encoding
- **Bootstrap CI**: Used in `evaluate_downstream.py` for aggregate metrics

## Notes

- **Overwrites previous results**: Each run replaces existing CSV files
- **Deterministic**: Results are reproducible with fixed random seeds
- **GPU recommended**: Significantly faster than CPU for model inference

## Troubleshooting

### Error: "Classification experiment not found"

**Problem**: Script cannot find exp1.1.1, exp1.1, or exp1 folders.

**Solution**:
```bash
cd mycode/classification
python reproduce_results.py
```

This will train all required experiments (exp0, exp1, exp1.1, exp1.1.1, exp2, exp3).

**Verification**:
```bash
ls output/  # Should show exp1, exp1.1, exp1.1.1 directories
ls output/exp1/data/  # Should contain mlb.pkl, y_val.npy, standard_scaler.pkl
```

### Error: "FileNotFoundError: mlb.pkl"

**Problem**: Classification experiment exists but is incomplete.

**Solution**: Re-run the classification experiment:
```bash
cd mycode/classification
python reproduce_results.py  # Will skip existing experiments by default
# Or force re-run for specific experiment:
# Edit reproduce_results.py to only include the problematic experiment
```

### Warning: "Skipped N diagnostic classes not in mlb"

**Explanation**: Some diagnostic classes from PTB-XL labels don't appear in the classification training set (likely filtered out due to insufficient samples).

**Expected behavior**: This is normal. The script skips these classes and continues with available classes.

**Not a problem** unless you see this for many common classes (NORM, MI, STTC, etc.).

### Memory Issues (CUDA out of memory)

**Problem**: GPU runs out of memory during prediction generation.

**Solutions**:
1. **Reduce batch size** in `denoise_12lead_signal()` call (currently 32)
2. **Evaluate fewer models**: Use `--classifiers fastai_xresnet1d101` instead of `--classifiers all`
3. **Process one diagnostic type at a time**: Modify code to process types sequentially with cache clearing

### Sampling Rate Mismatch Warning

**Problem**: Denoising models trained at different Hz than classification models.

**Example**: Denoising at 500Hz, classification at 100Hz.

**Solution**: Ensure consistency:
- Either retrain denoising models at 100Hz
- Or retrain classification models at 500Hz
- Or resample data appropriately

### No Output CSVs Generated

**Checklist**:
1. Check noise_configs in YAML: Must have valid paths in `qui_plot` section
2. Verify denoising models exist in experiment folder
3. Check classification models exist in experiment folders
4. Look for error messages in console output
5. Verify validation fold has data: Check `val_fold` in denoising config

### Old Results (Using exp0)

**Problem**: CSVs still show exp0 results instead of exp1.1.1/exp1.1/exp1.

**Solution**: The script now defaults to `use_diagnostic_specific_exps=True`. To verify:
- Check console output for "Using classification experiment: exp1.1.1" messages
- Check summary output for experiment names
- Delete old CSVs and re-run to force regeneration
- **Parallel processing**: Models are processed sequentially to manage memory
- **Class filtering**: Records without labels are automatically excluded
