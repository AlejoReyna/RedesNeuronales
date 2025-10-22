# Quick Start Guide - Horno Prediction Project

## Step-by-Step Instructions

### 1. Prepare Your Environment

```bash
# Navigate to the project directory
cd /Users/alberto/Documents/redes_neuronales/HornoPrediction

# Create a virtual environment (recommended)
gi
# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Add Your Data

Place your `Variables_Horno.csv` file in the `data/` directory:

```bash
# Make sure your data file is in the correct location
ls data/Variables_Horno.csv
```

**Important**: The target variable (POWER ON) should be in the **first column** (Column A) of the CSV file.

### 3. Run Exploratory Data Analysis (EDA)

```bash
# Option 1: Get instructions
python main.py --eda

# Option 2: Run the Jupyter notebook directly
cd notebooks
jupyter notebook eda_visualization.ipynb
```

The EDA notebook will help you:
- Understand your data distribution
- Identify correlations with the target variable
- Detect outliers and missing values
- Select the most important features

### 4. Train the Model

```bash
# Return to the project root (if you were in notebooks/)
cd ..

# Train the CNN model
python main.py --train
```

This will:
- Preprocess the data (clean, scale, split 70/30)
- Build the CNN architecture
- Train for up to 100 epochs (with early stopping)
- Save the trained model to `models/cnn_model.h5`
- Generate training curves in `results/training_curves.png`

**Expected time**: 5-15 minutes depending on your hardware and data size.

### 5. Evaluate the Model

```bash
# Evaluate on the test set
python main.py --evaluate
```

This will:
- Load the trained model
- Make predictions on the test set (last 30% of data)
- Calculate performance metrics
- Generate visualizations
- Save results to `results/metrics.json`

### 6. View Results

Check the generated files:

```bash
# View metrics
cat results/metrics.json

# View training curves
open results/training_curves.png  # On macOS
# or
xdg-open results/training_curves.png  # On Linux

# View confusion matrix (if classification task)
open results/confusion_matrix.png  # On macOS
```

## One-Command Complete Pipeline

Run training and evaluation together:

```bash
python main.py --all
```

## Typical Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place data file
cp /path/to/Variables_Horno.csv data/

# 3. Explore the data (optional but recommended)
cd notebooks && jupyter notebook eda_visualization.ipynb

# 4. Train and evaluate
cd .. && python main.py --all

# 5. Check results
cat results/metrics.json
```

## File Locations After Running

```
HornoPrediction/
├── data/
│   └── Variables_Horno.csv       # Your input data
├── models/
│   ├── cnn_model.h5              # Trained model
│   ├── cnn_model_best.h5         # Best model from training
│   └── scaler.pkl                # Feature scaler
├── results/
│   ├── metrics.json              # Performance metrics
│   ├── training_curves.png       # Loss/accuracy plots
│   └── confusion_matrix.png      # Confusion matrix (if applicable)
└── *.log                         # Log files
```

## Troubleshooting

### Issue: "Data file not found"
**Solution**: Copy `Variables_Horno.csv` to the `data/` directory

### Issue: "Model not found"
**Solution**: Run training first with `python main.py --train`

### Issue: Import errors
**Solution**: Make sure you're in the project root directory and virtual environment is activated

### Issue: Out of memory
**Solution**: Edit `src/train_cnn.py` and reduce the batch size from 32 to 16

## Command Reference

```bash
python main.py --eda          # Show EDA instructions
python main.py --preprocess   # Run preprocessing only
python main.py --train        # Train the model
python main.py --evaluate     # Evaluate the model
python main.py --all          # Run train + evaluate
```

## Expected Output Example

When you run evaluation, you should see something like:

```
============================================================
                     MODEL METRICS
============================================================
accuracy............................................ 0.945000
precision........................................... 0.940000
recall.............................................. 0.950000
f1_score............................................ 0.945000
mae................................................. 0.125000
mse................................................. 0.045000
rmse................................................ 0.212132
test_samples........................................ 3000
============================================================
```

## Next Steps

After getting your results:

1. Review the training curves to check for overfitting
2. Analyze the confusion matrix (for classification)
3. Compare metrics with baseline models
4. Consider hyperparameter tuning if needed
5. Document your findings for the course presentation

## Support

For detailed information, see the main `README.md` file.

Good luck with your project!


