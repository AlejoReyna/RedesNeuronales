"""
Utility functions for the Horno Prediction Project.
Includes logging setup, visualization functions, and helper utilities.
"""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime


def setup_logging(log_file='project.log'):
    """
    Configure logging for the project.
    
    Args:
        log_file (str): Path to the log file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_training_curves(history, output_path='results/training_curves.png'):
    """
    Plot and save training and validation curves.
    
    Args:
        history: Keras History object from model.fit()
        output_path (str): Path to save the plot
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy or MAE curve
    metric_key = 'accuracy' if 'accuracy' in history.history else 'mae'
    metric_label = 'Accuracy' if metric_key == 'accuracy' else 'MAE'
    
    axes[1].plot(history.history[metric_key], label=f'Training {metric_label}', linewidth=2)
    if f'val_{metric_key}' in history.history:
        axes[1].plot(history.history[f'val_{metric_key}'], label=f'Validation {metric_label}', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel(metric_label, fontsize=12)
    axes[1].set_title(f'Model {metric_label} Over Epochs', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Training curves saved to {output_path}")


def plot_confusion_matrix(y_true, y_pred, output_path='results/confusion_matrix.png', threshold=0.5):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities or labels
        output_path (str): Path to save the plot
        threshold (float): Threshold for binary classification
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert predictions to binary if needed
    if y_pred.ndim > 1:
        y_pred_binary = (y_pred > threshold).astype(int).flatten()
    else:
        y_pred_binary = (y_pred > threshold).astype(int)
    
    y_true_binary = y_true.astype(int).flatten()
    
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                square=True, linewidths=1, linecolor='black')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Confusion matrix saved to {output_path}")


def save_metrics_to_json(metrics_dict, output_path='results/metrics.json'):
    """
    Save metrics dictionary to JSON file.
    
    Args:
        metrics_dict (dict): Dictionary containing metrics
        output_path (str): Path to save the JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add timestamp
    metrics_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(output_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    logging.info(f"Metrics saved to {output_path}")


def print_metrics(metrics_dict):
    """
    Pretty print metrics to console.
    
    Args:
        metrics_dict (dict): Dictionary containing metrics
    """
    print("\n" + "="*60)
    print(" "*20 + "MODEL METRICS")
    print("="*60)
    
    for key, value in metrics_dict.items():
        if key != 'timestamp':
            if isinstance(value, float):
                print(f"{key:.<40} {value:.6f}")
            else:
                print(f"{key:.<40} {value}")
    
    print("="*60 + "\n")


def ensure_directory_structure():
    """
    Ensure all required directories exist.
    """
    directories = ['data', 'notebooks', 'src', 'models', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logging.info("Directory structure verified")


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    import random
    random.seed(seed)
    
    logging.info(f"Random seeds set to {seed}")


