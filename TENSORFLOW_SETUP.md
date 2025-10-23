# TensorFlow Setup Guide

## Issue
The project uses TensorFlow for deep learning, but TensorFlow is not compatible with Python 3.14 (the current Python version on your system).

## ✅ Solution: Use Python 3.13 (Working!)
TensorFlow now supports Python 3.13! The environment is already set up and working:

```bash
# Navigate to project directory
cd /Users/alexis/Documents/RedesNeuronales/RedesNeuronales

# Activate the Python 3.13 environment
source venv311/bin/activate

# Run the project
python3 main.py --train
```

### Alternative: Use Current Environment (Limited Functionality)
The current setup allows you to run EDA and preprocessing without TensorFlow:

```bash
# Activate current environment
source venv/bin/activate

# Run EDA (works)
python3 main.py --eda

# Run preprocessing (works)
python3 main.py --preprocess

# Run training (shows TensorFlow error)
python3 main.py --train
```

## What Works Now

### With venv311 (Python 3.13):
- ✅ EDA command (`python3 main.py --eda`)
- ✅ Preprocessing command (`python3 main.py --preprocess`)
- ✅ Training command (`python3 main.py --train`)
- ✅ Evaluation command (`python3 main.py --evaluate`)
- ✅ All data analysis and visualization

### With venv (Python 3.14):
- ✅ EDA command (`python3 main.py --eda`)
- ✅ Preprocessing command (`python3 main.py --preprocess`)
- ❌ Training command (requires TensorFlow)
- ❌ Evaluation command (requires TensorFlow)

## Commands Summary

**For full functionality:**
```bash
cd /Users/alexis/Documents/RedesNeuronales/RedesNeuronales
source venv311/bin/activate
python3 main.py --train
```

**For data analysis only:**
```bash
cd /Users/alexis/Documents/RedesNeuronales/RedesNeuronales
source venv/bin/activate
python3 main.py --eda
python3 main.py --preprocess
```

## Technical Details
The code has been modified to:
- Import TensorFlow-dependent modules only when available
- Show helpful error messages when TensorFlow is missing
- Allow core functionality (EDA, preprocessing) to work without TensorFlow
