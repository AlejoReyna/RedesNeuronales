"""
Source package for Horno Prediction Project.
Contains modules for preprocessing, training, evaluation, and utilities.
"""

__version__ = '1.0.0'
__author__ = 'Horno Prediction Team'

from src.preprocess import DataPreprocessor
from src import utils

# Import TensorFlow-dependent modules only when needed
try:
    from src.train_cnn import CNNModel
    from src.evaluate import ModelEvaluator
    _tensorflow_available = True
except ImportError:
    _tensorflow_available = False
    CNNModel = None
    ModelEvaluator = None

__all__ = [
    'DataPreprocessor',
    'CNNModel',
    'ModelEvaluator',
    'utils',
    '_tensorflow_available'
]


