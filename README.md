# Industrial Oven Power Prediction using CNN

A deep learning solution for predicting industrial oven power status using Convolutional Neural Networks (CNN). This project implements a complete machine learning pipeline including data preprocessing, model training, evaluation, and reporting.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Model Architecture](#model-architecture)
- [Data](#data)
- [Output Files](#output-files)

## Overview

This project applies deep learning techniques to predict the POWER ON status of industrial ovens based on operational variables. The solution includes:

- **Data Preprocessing**: Automated data cleaning, scaling, and temporal train-test splitting
- **CNN Model**: Multi-layer 1D convolutional neural network optimized for time-series prediction
- **Evaluation**: Comprehensive performance metrics and visualizations
- **Reporting**: Automated PDF report generation with training curves and performance analysis

The system supports both classification and regression tasks, automatically detecting the appropriate task type based on the target variable distribution.

## Project Structure

```
RedesNeuronales/
│
├── data/
│   └── Variables_Horno.csv         # Input dataset (industrial oven variables)
│
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── utils.py                    # Utility functions (logging, plotting, metrics)
│   ├── preprocess.py               # Data preprocessing pipeline
│   ├── train_cnn.py                # CNN model architecture and training
│   ├── evaluate.py                 # Model evaluation and metrics
│   └── report.py                   # PDF report generation
│
├── models/                         # Trained models and scalers (generated)
│   ├── cnn_model.h5
│   ├── cnn_model_best.h5
│   └── scaler.pkl
│
├── results/                        # Output results (generated)
│   ├── metrics.json
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── report.pdf
│
├── notebooks/
│   └── eda_visualization.ipynb     # Exploratory Data Analysis
│
├── main.py                         # Main CLI interface
├── data_leakage_analysis.py        # Data validation script
├── robustness_tests.py             # Model robustness testing
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Requirements

- Python 3.8 or higher
- TensorFlow 2.13+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- Jupyter (for EDA notebook)

See `requirements.txt` for complete dependency list.

## Installation

1. **Clone or download the repository**

```bash
cd RedesNeuronales
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Verify data file**

Ensure `Variables_Horno.csv` is present in the `data/` directory.

## Usage

The project provides a command-line interface through `main.py` with several options:

### Complete Pipeline (Recommended)

Run the entire pipeline (training + evaluation):

```bash
python main.py --all
```

### Individual Steps

**1. Exploratory Data Analysis**

```bash
python main.py --eda
```

Then open the Jupyter notebook:

```bash
cd notebooks
jupyter notebook eda_visualization.ipynb
```

The EDA notebook provides:
- Data quality checks (missing values, duplicates)
- Statistical analysis and distributions
- Correlation analysis with target variable
- Feature importance visualization
- Outlier detection

**2. Data Preprocessing Only**

```bash
python main.py --preprocess
```

Performs:
- Data loading and validation
- Missing value imputation
- Feature scaling (StandardScaler)
- Temporal 70/30 train-test split

**3. Model Training**

```bash
python main.py --train
```

Trains the CNN model with:
- Early stopping (patience: 10 epochs)
- Model checkpointing (saves best model)
- Learning rate reduction on plateau
- Batch normalization and dropout for regularization

Training typically takes 5-15 minutes depending on hardware.

**4. Model Evaluation**

```bash
python main.py --evaluate
```

Generates:
- Classification/regression metrics
- Confusion matrix (for classification)
- Performance report in JSON format
- PDF report with visualizations

### Advanced Options

**Feature Selection**

Select top-k most correlated features:

```bash
python main.py --all --top_k 10
```

Select features above correlation threshold:

```bash
python main.py --all --corr_threshold 0.3
```

**Baseline Comparison**

Compare with a baseline model:

```bash
python main.py --evaluate --baseline path/to/baseline_model.pkl
```

**Generate Report from Existing Results**

```bash
python main.py --report
```

### Additional Analysis Scripts

**Data Leakage Analysis**

Verify data integrity and check for potential data leakage:

```bash
python data_leakage_analysis.py
```

**Robustness Testing**

Test model performance with different validation strategies:

```bash
python robustness_tests.py
```

This script performs:
- Random shuffled split validation
- K-fold cross-validation
- Feature ablation tests

## Results

### Performance Metrics

The evaluation generates comprehensive metrics including:

**For Classification Tasks:**
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

**For Regression Tasks:**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Sample Output

After running the complete pipeline, you'll see output similar to:

```
============================================================
                     MODEL METRICS
============================================================
accuracy............................................ 0.9450
precision........................................... 0.9400
recall.............................................. 0.9500
f1_score............................................ 0.9450
mae................................................. 0.1250
mse................................................. 0.0450
rmse................................................ 0.2121
test_samples........................................ 300
============================================================
```

## Model Architecture

The CNN model consists of:

### Architecture Layers

1. **Input Layer**: Accepts preprocessed feature vectors
2. **Convolutional Blocks** (3 blocks):
   - Conv1D layers (32, 64, 128 filters)
   - Batch Normalization
   - ReLU activation
   - MaxPooling1D
   - Dropout (0.3)
3. **Flattening Layer**
4. **Dense Layers**:
   - Dense(128) with ReLU and Dropout(0.4)
   - Dense(64) with ReLU and Dropout(0.3)
5. **Output Layer**:
   - Classification: Dense(1) with sigmoid
   - Regression: Dense(1) linear

### Training Configuration

- **Optimizer**: Adam
- **Loss Function**: 
  - Binary crossentropy (classification)
  - Mean squared error (regression)
- **Batch Size**: 32
- **Max Epochs**: 100
- **Validation Split**: 15%
- **Early Stopping**: Patience of 10 epochs

## Data

### Input Format

The system expects a CSV file (`Variables_Horno.csv`) with:
- **Target Variable**: First column (POWER ON status)
- **Features**: Remaining columns (operational variables)
- **Format**: Comma-separated values with header row

### Data Split

- **Training Set**: First 70% of data (temporal split)
- **Test Set**: Last 30% of data
- **Validation**: 15% of training data for validation during training

The temporal split maintains the time-series nature of the data, which is crucial for realistic performance evaluation in industrial settings.

## Output Files

After running the complete pipeline, the following files are generated:

### Models Directory (`models/`)

- `cnn_model.h5`: Final trained model
- `cnn_model_best.h5`: Best model checkpoint during training
- `scaler.pkl`: Fitted StandardScaler for feature normalization

### Results Directory (`results/`)

- `metrics.json`: Performance metrics in JSON format
- `training_curves.png`: Training and validation loss/accuracy curves
- `confusion_matrix.png`: Confusion matrix visualization (classification only)
- `report.pdf`: Comprehensive PDF report with all results
- `correlation_heatmap.png`: Feature correlation analysis (from data_leakage_analysis.py)
- `class_distribution.png`: Target variable distribution (from robustness_tests.py)

### Log Files (Root Directory)

- `preprocessing.log`: Data preprocessing logs
- `training.log`: Model training logs
- `evaluation.log`: Evaluation process logs

## Troubleshooting

### Common Issues

**Issue**: `Data file not found`  
**Solution**: Ensure `Variables_Horno.csv` is in the `data/` directory

**Issue**: `Model not found during evaluation`  
**Solution**: Run training first with `python main.py --train`

**Issue**: `Out of memory error`  
**Solution**: Reduce batch size in `src/train_cnn.py` (line with `batch_size=32`)

**Issue**: `Import errors`  
**Solution**: Ensure you're in the project root directory and virtual environment is activated

## Development

### Running Tests

```bash
# Data integrity checks
python data_leakage_analysis.py

# Model robustness tests
python robustness_tests.py
```

### Modifying the Model

Edit `src/train_cnn.py` to modify:
- Number of convolutional layers
- Filter sizes and counts
- Dropout rates
- Dense layer dimensions

### Custom Preprocessing

Edit `src/preprocess.py` to modify:
- Feature scaling method
- Train-test split ratio
- Feature selection criteria

## License

This project is provided as-is for academic and research purposes.

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.

---

**Last Updated**: October 2025  
**Python Version**: 3.8+  
**TensorFlow Version**: 2.13+

