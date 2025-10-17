# ECG PTB-XL Benchmarking Project

This repository contains code for ECG classification and denoising experiments on the PTB-XL dataset.

## Project Structure

```
├── code/                           # All source code
│   ├── classification/             # ECG classification experiments
│   │   ├── configs/               # Model configurations
│   │   ├── experiments/           # Experiment runners (SCP, noise robustness)
│   │   ├── models/                # Neural network models
│   │   ├── utils/                 # Utility functions
│   │   ├── output/                # Classification experiment outputs
│   │   ├── reproduce_results.py   # Script to reproduce benchmark results
│   │   ├── run_noise_experiments.py  # Run noise robustness experiments
│   │   ├── README.md              # Classification documentation
│   │   └── NOISE_EXPERIMENT_README.md  # Noise experiment documentation
│   │
│   ├── denoising/                 # ECG denoising experiments
│   │   ├── configs/               # Denoising configurations
│   │   │   └── denoising_config.yaml
│   │   ├── denoising_utils/       # Denoising utilities (preprocessing, training)
│   │   ├── denoising_models/      # Denoising model architectures
│   │   │   └── ECG-processing/    # Model implementations (FCN, UNet, IMUnet, DRnet)
│   │   ├── output/                # Denoising experiment outputs
│   │   ├── train.py               # Training script
│   │   ├── evaluate_similarity.py # Evaluation: signal similarity metrics
│   │   ├── evaluate_downstream.py # Evaluation: downstream classification
│   │   ├── 0_README.md            # Denoising documentation
│   │   └── 0_downstream_README.md # Downstream evaluation documentation
│   │
│   ├── ecg_noise/                 # ECG noise generation package
│   ├── eda/                       # Exploratory data analysis
│   │
├── data/                          # PTB-XL dataset
│   └── ptbxl/
│
├── noise/                         # Noise experiment data and configs
│   ├── config/
│   └── data/
│
├── notebooks/                     # Jupyter notebooks
│   └── selection_denoising.ipynb
│
├── output/                        # Root-level outputs (legacy/shared)
│
├── tests/                         # Unit tests
│
└── environment_setup.md           # Environment setup instructions
```

## Quick Start

### 1. Setup Environment

Follow instructions in `environment_setup.md` to set up the Python environment.

### 2. Download Data

```bash
bash get_datasets.sh
```

### 3. Run Classification Experiments

```bash
cd code/classification
python reproduce_results.py
```

See `code/classification/README.md` for more details.

### 4. Run Denoising Experiments

```bash
cd code/denoising
python train.py --config configs/denoising_config.yaml
```

See `code/denoising/0_README.md` for more details.

### 5. Run Noise Robustness Experiments

```bash
cd code/classification
python run_noise_experiments.py
```

See `code/classification/NOISE_EXPERIMENT_README.md` for more details.

## Key Components

### Classification (`code/classification/`)
- Benchmark ECG classification models on PTB-XL
- Multiple architectures: ResNet, Inception, LSTM, FCN, etc.
- SCP statement classification experiments
- Noise robustness evaluation

### Denoising (`code/denoising/`)
- Train denoising models for ECG signals
- Evaluate denoising quality (SNR, RMSE)
- Test impact on downstream classification tasks
- Support for multiple noise types (BW, EM, MA)

### ECG Noise (`code/ecg_noise/`)
- Synthetic ECG noise generation
- Multiple noise types: Baseline Wander (BW), Electrode Motion (EM), Muscle Artifact (MA)
- Configurable noise levels and characteristics

### EDA (`code/eda/`)
- Exploratory data analysis notebooks and scripts
- Data quality assessment
- Statistical analysis of ECG signals

## Imports and Module Structure

After restructuring, all imports follow the pattern:
```python
from code.classification.models import model_name
from code.classification.utils.utils import function_name
from code.denoising.denoising_utils.preprocessing import preprocess_function
```

## Configuration Files

- Classification configs: `code/classification/configs/`
- Denoising config: `code/denoising/configs/denoising_config.yaml`
- Noise configs: `code/ecg_noise/configs/`

## Documentation

- Main classification README: `code/classification/README.md`
- Denoising README: `code/denoising/0_README.md`
- Noise experiments: `code/classification/NOISE_EXPERIMENT_README.md`
- Downstream evaluation: `code/denoising/0_downstream_README.md`

## License

See `LICENSE` file for details.
