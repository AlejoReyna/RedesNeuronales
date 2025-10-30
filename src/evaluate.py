"""
Model Evaluation Module for Horno Prediction Project.
Evaluates trained CNN model and generates performance metrics and visualizations.
"""

import os
import json
import logging
import numpy as np
import joblib
import pickle
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report
)
from src.preprocess import DataPreprocessor
from src.utils import (
    setup_logging, plot_confusion_matrix, 
    save_metrics_to_json, print_metrics, set_random_seeds
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Handles model evaluation and metrics generation.
    """
    
    def __init__(self, model_path='models/cnn_model.h5', scaler_path='models/scaler.pkl', target_scaler_path='models/target_scaler.pkl', baseline_path=None):
        """
        Initialize the evaluator.
        
        Args:
            model_path (str): Path to saved model
            scaler_path (str): Path to saved scaler
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.target_scaler_path = target_scaler_path
        self.target_scaler = None
        self.task_type = None
        self.baseline_path = baseline_path
        self.baseline_model = None
        self.target_transform = None
        
    def load_model_and_scaler(self):
        """
        Load the trained model and scaler.
        """
        logger.info(f"Loading model from {self.model_path}")
        try:
            # Try loading with custom objects for metrics compatibility
            self.model = keras.models.load_model(
                self.model_path,
                custom_objects={
                    'mse': keras.losses.MeanSquaredError(),
                    'mae': keras.metrics.MeanAbsoluteError(),
                    'rmse': keras.metrics.RootMeanSquaredError(),
                },
                compile=False
            )
            # Recompile if needed
            if not self.model.built:
                self.model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['mae', 'mse']
                )
        except Exception as e:
            logger.warning(f"Error loading model with custom objects: {e}")
            # Fallback: try loading without custom objects
            try:
                self.model = keras.models.load_model(self.model_path, compile=False)
                self.model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['mae', 'mse']
                )
            except Exception as e2:
                logger.error(f"Failed to load model: {e2}")
                raise
        
        logger.info(f"Loading scaler from {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)
        
        logger.info("Model and scaler loaded successfully")

        if self.target_scaler_path and os.path.exists(self.target_scaler_path):
            try:
                self.target_scaler = joblib.load(self.target_scaler_path)
                logger.info(f"Target scaler loaded from {self.target_scaler_path}")
                meta_path = self.target_scaler_path.replace('.pkl', '_meta.json')
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as meta_file:
                        meta = json.load(meta_file)
                        self.target_transform = meta.get('target_transform')
            except Exception as exc:
                logger.warning(f"Could not load target scaler: {exc}")

        # Try to load baseline model if provided
        if self.baseline_path and os.path.exists(self.baseline_path):
            try:
                logger.info(f"Loading baseline model from {self.baseline_path}")
                with open(self.baseline_path, 'rb') as f:
                    self.baseline_model = pickle.load(f)
                logger.info("Baseline model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load baseline model: {e}")
    
    def detect_task_type(self, y_true):
        """
        Detect whether this is a classification or regression task.
        
        Args:
            y_true (np.ndarray): True labels
            
        Returns:
            str: 'classification' or 'regression'
        """
        unique_values = np.unique(y_true)
        
        if len(unique_values) <= 10 and set(unique_values).issubset(set(range(10))):
            self.task_type = 'classification'
        else:
            self.task_type = 'regression'
        
        logger.info(f"Task type: {self.task_type}")
        return self.task_type
    
    def predict(self, X, already_scaled=False):
        """
        Make predictions on input data.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predictions
        """
        logger.info(f"Making predictions on {len(X)} samples")
        if not already_scaled:
            if self.scaler is None:
                raise ValueError("Scaler not loaded. Please call load_model_and_scaler() first.")
            X = self.scaler.transform(X)
        predictions = self.model.predict(X, verbose=0)
        if self.task_type == 'regression':
            if self.target_scaler is not None:
                predictions = self.target_scaler.inverse_transform(predictions)
            if self.target_transform == 'log1p':
                predictions = np.expm1(predictions)
        predictions = predictions.reshape(-1, 1) if predictions.ndim == 1 else predictions
        if predictions.shape[1] == 1:
            predictions = predictions.flatten()
        return predictions
    
    def calculate_classification_metrics(self, y_true, y_pred, threshold=0.5):
        """
        Calculate classification metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted probabilities
            threshold (float): Classification threshold
            
        Returns:
            dict: Dictionary of metrics
        """
        y_pred_binary = (y_pred > threshold).astype(int).flatten()
        y_true_binary = y_true.astype(int).flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
            'threshold': threshold
        }
        
        # Also calculate regression-style metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        return metrics
    
    def calculate_regression_metrics(self, y_true, y_pred):
        """
        Calculate regression metrics.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            dict: Dictionary of metrics
        """
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        metrics = {
            'mae': mean_absolute_error(y_true_flat, y_pred_flat),
            'mse': mean_squared_error(y_true_flat, y_pred_flat),
            'rmse': np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)),
            'r2_score': r2_score(y_true_flat, y_pred_flat),
            'mean_absolute_percentage_error': np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-10))) * 100
        }
        
        return metrics
    
    def evaluate(self, X_test, y_test, X_test_scaled=None):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (np.ndarray): Test features (can be unscaled if scaler is available)
            y_test (np.ndarray): Test targets
            X_test_scaled (np.ndarray, optional): Pre-scaled test features. If None, will scale using self.scaler
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info("=" * 60)
        logger.info("STARTING MODEL EVALUATION")
        logger.info("=" * 60)
        
        # Detect task type
        self.detect_task_type(y_test)
        
        # Scale features if not already scaled
        if X_test_scaled is None:
            if self.scaler is None:
                raise ValueError("Scaler not loaded. Please call load_model_and_scaler() first.")
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test_scaled
        
        # Make predictions (already scaled features)
        y_pred = self.predict(X_test_scaled, already_scaled=True)
        
        # Calculate metrics based on task type
        if self.task_type == 'classification':
            metrics = self.calculate_classification_metrics(y_test, y_pred)
            
            # Generate confusion matrix
            plot_confusion_matrix(y_test, y_pred, 'results/confusion_matrix.png')
            
        else:
            metrics = self.calculate_regression_metrics(y_test, y_pred)
        
        # Add sample statistics
        metrics['test_samples'] = len(y_test)
        metrics['target_mean'] = float(np.mean(y_test))
        metrics['target_std'] = float(np.std(y_test))
        metrics['prediction_mean'] = float(np.mean(y_pred))
        metrics['prediction_std'] = float(np.std(y_pred))
        
        # Evaluate baseline if available
        if self.baseline_model is not None:
            try:
                logger.info("Evaluating baseline model")
                baseline_pred = self.baseline_model.predict(X_test)
                if self.task_type == 'classification':
                    base_metrics = self.calculate_classification_metrics(y_test, baseline_pred)
                else:
                    base_metrics = self.calculate_regression_metrics(y_test, baseline_pred)
                for k, v in base_metrics.items():
                    metrics[f"baseline_{k}"] = v
            except Exception as e:
                logger.warning(f"Baseline evaluation failed: {e}")

        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETED")
        logger.info("=" * 60)
        
        return metrics
    
    def generate_detailed_report(self, y_test, y_pred):
        """
        Generate a detailed classification report.
        
        Args:
            y_test (np.ndarray): True labels
            y_pred (np.ndarray): Predicted probabilities
        """
        if self.task_type == 'classification':
            y_pred_binary = (y_pred > 0.5).astype(int).flatten()
            y_test_binary = y_test.astype(int).flatten()
            
            print("\nDetailed Classification Report:")
            print("=" * 60)
            print(classification_report(y_test_binary, y_pred_binary))
            print("=" * 60)


def main():
    """
    Main function for model evaluation.
    """
    # Set random seeds
    set_random_seeds(42)
    
    # Setup logging
    setup_logging('evaluation.log')
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path='models/cnn_model.h5',
        scaler_path='models/scaler.pkl'
    )
    
    # Load model and scaler
    evaluator.load_model_and_scaler()
    
    # Evaluate model
    metrics = evaluator.evaluate(X_test, y_test)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save metrics to JSON
    save_metrics_to_json(metrics, 'results/metrics.json')
    
    # Generate predictions for detailed report
    y_pred = evaluator.predict(X_test)
    evaluator.generate_detailed_report(y_test, y_pred)
    
    logger.info("Evaluation pipeline completed successfully")
    
    # Print file locations
    print("\nGenerated files:")
    print(f"  - Model: models/cnn_model.h5")
    print(f"  - Scaler: models/scaler.pkl")
    print(f"  - Metrics: results/metrics.json")
    print(f"  - Training curves: results/training_curves.png")
    if evaluator.task_type == 'classification':
        print(f"  - Confusion matrix: results/confusion_matrix.png")


if __name__ == '__main__':
    main()

