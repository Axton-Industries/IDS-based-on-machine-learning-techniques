# IDS-based-on-machine-learning-techniques

An Intrusion Detection System (IDS) implementation leveraging various machine learning techniques to detect network intrusions and anomalies in traffic data. This project implements a complete ML pipeline from data loading and preprocessing to model training, hyperparameter tuning, and evaluation.

## Overview

This project explores both supervised and unsupervised machine learning approaches for network intrusion detection:
- **Supervised learning**: XGBoost and Random Forest classifiers for binary (Benign vs Attack) and multiclass attack detection
- **Unsupervised learning**: Isolation Forest and Autoencoder for anomaly detection
- **Evaluation**: Comprehensive metrics including ROC-AUC, PR-AUC, confusion matrices, and classification reports with operational considerations (e.g., 1% FPR threshold)

The project is built around the CICIDS (Canadian Institute for Cybersecurity Intrusion Detection System) style datasets with network flow features.

## Project Structure

```
IDS-based-on-machine-learning-techniques/
├── data/
│   ├── processed/          # Preprocessed and engineered data (parquet files)
│   └── raw/                # Raw CSV datasets from network captures
├── notebooks/              # Jupyter notebooks for the full ML pipeline
│   ├── 01_data_loading.ipynb      # Load and merge raw CSV files
│   ├── 02_preprocessing.ipynb     # Data cleaning, downcasting, handling types
│   ├── 03_eda.ipynb               # Exploratory data analysis
│   ├── 04_feature_engineering.ipynb # Train/test split, scaling, encoding
│   └── 05_modeling.ipynb          # Model training, tuning, and evaluation
├── src/
│   ├── data/               # Data processing utilities
│   │   ├── cat_encoding.py      # Custom encoders for categorical features
│   │   ├── downcast.py          # Memory optimization via numeric downcasting
│   │   ├── load.py              # CSV loading and merging utilities
│   │   ├── log_transform.py     # Log transformation for skewed features
│   │   └── var_type.py          # Feature type classification (symbolic/continuous)
│   ├── model/              # Model training and evaluation utilities
│   │   ├── evaluation.py        # Evaluation metrics, plots, thresholding
│   │   └── tunning.py           # Hyperparameter tuning with Optuna
│   └── __init__.py
├── anaconda_projects/      # Conda environment/project files
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

## Features

### Data Processing (`src/data/`)
- **Merging**: `load_and_merge_csvs()` - Combines multiple CSV files from a directory
- **Type classification**: `split_symbolic_continuous()` - Automatically splits features into symbolic and continuous
- **Memory optimization**: Intelligent downcasting of numeric types (int32/uint32/float32) based on value ranges and precision
- **Log transformation**: `shift_log_transform()` - Handles skewed distributions with shift+log1p
- **Custom encoding**: `CustomBenignFirstEncoder()` - Preserves "Benign" as first class for interpretable labeling

### Model Training & Evaluation (`src/model/`)
- **Evaluation**: Functions for binary and multiclass evaluation with ROC curves, PR curves, confusion matrices, and classification reports
- **Threshold optimization**: `get_defensive_threshold()` - F-beta based threshold tuning on validation set to minimize false negatives (beta=2 for defense-prioritizing scenarios)
- **Hyperparameter tuning**: Optuna-based tuning for:
  - **XGBoost** (binary & multiclass): Learning rate, depth, regularization (alpha, lambda, gamma), subsampling
  - **Random Forest** (binary & multiclass): Tree count, depth, feature sampling, split criteria
  - **Isolation Forest** (unsupervised): Anomaly detection with contamination tuning
  - **Autoencoder** (unsupervised): Deep neural network with custom architecture, dropout, L2 regularization
- **GPU support**: XGBoost and TensorFlow configured for CUDA acceleration where available

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for accelerated training)

### Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/IDS-based-on-machine-learning-techniques.git
cd IDS-based-on-machine-learning-techniques
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. (Optional) Create a dedicated environment
```bash
conda env create -f anaconda_projects/environment.yml  # if available
```

## Usage

The project follows a sequential notebook workflow. Start from notebook 01 and proceed through 05:

| Notebook | Purpose |
|---|---|
| `01_data_loading.ipynb` | Load all CSV files from `data/raw/`, merge them into a unified dataset, and save as parquet |
| `02_preprocessing.ipynb` | Clean data, handle mixed types, downcast for memory efficiency, prepare for analysis |
| `03_eda.ipynb` | Exploratory data analysis - distributions, class balance, correlations, attack patterns |
| `04_feature_engineering.ipynb` | Split into train/validation/test sets, encode categorical features, scale numeric features, handle class imbalance |
| `05_modeling.ipynb` | Train models (XGBoost, RF, Isolation Forest, Autoencoder), tune hyperparameters with Optuna, evaluate performance |

### Using the Python modules directly

```python
# Data loading
from data.load import load_and_merge_csvs
df = load_and_merge_csvs("data/raw")

# Feature type splitting
from data.var_type import split_symbolic_continuous
sym, cont = split_symbolic_continuous(df)

# Memory optimization
from data.downcast import downcast_numeric_columns
downcast_numeric_columns(df)

# Hyperparameter tuning
from model.tunning import tune_xgboost
best_params = tune_xgboost(X_train, y_train, scale_pos_weight=scale_pos_weight, n_trials=30)

# Evaluation
from model.evaluation import evaluate_binary, get_defensive_threshold
threshold = get_defensive_threshold(val_scores, y_val, beta=2)
evaluate_binary(test_scores, y_test, threshold)
```

## Data

This project expects network flow datasets in CSV format placed in `data/raw/`. The code is designed to handle multiple CSV files (e.g., different days/sessions) which are concatenated. Common features include:
- Flow metadata: Source/Destination ports, Protocol, Flow Duration
- Packet statistics: Forward/Backward packet counts, lengths, IAT (Inter-Arrival Time)
- Flag counts: FIN, SYN, RST, PSH, ACK, URG, etc.
- Window sizes, active/idle times
- Label: Traffic class (Benign/Attack with specific attack types)

Preprocessed outputs are saved as Parquet files in `data/processed/` for efficient I/O.

## Key Design Choices

- **Memory efficiency**: Aggressive downcasting with validation to prevent precision loss
- **Defensive security posture**: F-beta thresholding (beta=2) prioritizes recall to minimize missed attacks
- **Reproducibility**: Fixed random seeds (42) throughout
- **GPU acceleration**: Enabled for XGBoost (hist + cuda) and TensorFlow when available
- **Robust evaluation**: Cross-validation for tuning, strict train/val/test separation with blind evaluation
- **Operational awareness**: ROC curves include 1% FPR operational limit reference line
- **Class imbalance handling**: `scale_pos_weight`, `class_weight='balanced'`, and stratified splits

## Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Classical ML algorithms (RF, IsolationForest, preprocessing, metrics)
- `xgboost` - Gradient boosting with GPU support
- `tensorflow` - Autoencoder implementation
- `optuna` - Hyperparameter optimization
- `matplotlib` - Visualization
- `jupyter` - Interactive notebooks

See `requirements.txt` for exact versions.

## License

This project is provided for educational and research purposes.
