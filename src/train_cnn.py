"""
CNN Model Training Module for Horno Prediction Project.
Builds and trains a 1D Convolutional Neural Network for POWER ON prediction.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.preprocess import DataPreprocessor
from src.utils import setup_logging, save_training_curves, set_random_seeds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNNModel:
    """
    Convolutional Neural Network for POWER ON prediction.
    """
    
    def __init__(self, input_shape, task_type='regression'):
        """
        Initialize the CNN model.
        
        Args:
            input_shape (tuple): Shape of input features
            task_type (str): 'regression' or 'classification'
        """
        self.input_shape = input_shape
        self.task_type = task_type
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build the CNN architecture.
        
        Returns:
            keras.Model: Compiled CNN model
        """
        logger.info("Building CNN model architecture")
        logger.info(f"Input shape: {self.input_shape}")
        logger.info(f"Task type: {self.task_type}")
        
        model = models.Sequential([
            # Reshape for Conv1D: (batch_size, timesteps, features)
            layers.Reshape((self.input_shape[0], 1), input_shape=self.input_shape),
            
            # First convolutional block
            layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            # Second convolutional block
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            # Third convolutional block
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Flatten(),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1, activation='sigmoid' if self.task_type == 'classification' else 'linear')
        ])
        
        self.model = model
        logger.info("Model architecture built successfully")
        
        return model
    
    def compile_model(self):
        """
        Compile the model with appropriate loss and metrics.
        """
        logger.info("Compiling model")
        
        if self.task_type == 'classification':
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 
                      tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.Recall(name='recall'),
                      tf.keras.metrics.AUC(name='auc')]
        else:
            loss = 'mse'
            metrics = ['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with loss: {loss}")
        
    def get_callbacks(self):
        """
        Configure training callbacks.
        
        Returns:
            list: List of Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/cnn_model_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.15):
        """
        Train the CNN model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of training data to use for validation
            
        Returns:
            History: Training history object
        """
        logger.info("=" * 60)
        logger.info("STARTING MODEL TRAINING")
        logger.info("=" * 60)
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Validation split: {validation_split}")
        
        callbacks = self.get_callbacks()
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 60)
        
        return self.history
    
    def save_model(self, model_path='models/cnn_model.h5'):
        """
        Save the trained model.
        
        Args:
            model_path (str): Path to save the model
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path='models/cnn_model.h5'):
        """
        Load a saved model.
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    def summary(self):
        """
        Print model summary.
        """
        if self.model is not None:
            self.model.summary()
        else:
            logger.warning("Model not built yet. Call build_model() first.")


def determine_task_type(y):
    """
    Determine if the task is classification or regression based on target values.
    
    Args:
        y (np.ndarray): Target values
        
    Returns:
        str: 'classification' or 'regression'
    """
    unique_values = np.unique(y)
    
    # If only 2 unique values and they are 0 and 1, it's binary classification
    if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
        logger.info("Task type detected: BINARY CLASSIFICATION")
        return 'classification'
    
    # If few unique values (< 10) and all integers, might be classification
    elif len(unique_values) < 10 and np.all(y == y.astype(int)):
        logger.info("Task type detected: MULTI-CLASS CLASSIFICATION")
        return 'classification'
    
    # Otherwise, it's regression
    else:
        logger.info("Task type detected: REGRESSION")
        return 'regression'


def main():
    """
    Main function for model training.
    """
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Setup logging
    setup_logging('training.log')
    
    # Preprocess data
    logger.info("Starting data preprocessing")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    
    # Determine task type
    task_type = determine_task_type(y_train)
    
    # Build and compile model
    input_shape = (X_train.shape[1],)
    cnn_model = CNNModel(input_shape=input_shape, task_type=task_type)
    cnn_model.build_model()
    cnn_model.compile_model()
    
    # Print model summary
    print("\nModel Architecture:")
    print("=" * 60)
    cnn_model.summary()
    print("=" * 60)
    
    # Train model
    history = cnn_model.train(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.15
    )
    
    # Save model
    cnn_model.save_model('models/cnn_model.h5')
    
    # Save training curves
    save_training_curves(history, 'results/training_curves.png')
    
    logger.info("Training pipeline completed successfully")


if __name__ == '__main__':
    main()


