"""
Main CLI Script for Horno Prediction Project.
Provides command-line interface for training, evaluating, and running EDA.
"""

import argparse
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import DataPreprocessor
from src.utils import (
    setup_logging, save_training_curves, set_random_seeds,
    ensure_directory_structure, print_metrics, save_metrics_to_json
)
from src.report import generate_pdf_report

# Import TensorFlow-dependent modules only when available
try:
    from src.train_cnn import CNNModel, determine_task_type
    from src.evaluate import ModelEvaluator
    _tensorflow_available = True
except ImportError:
    _tensorflow_available = False
    CNNModel = None
    ModelEvaluator = None
    determine_task_type = None


def run_eda():
    """
    Run exploratory data analysis.
    Opens the Jupyter notebook for EDA.
    """
    print("\n" + "=" * 60)
    print(" " * 15 + "RUNNING EDA")
    print("=" * 60)
    print("\nTo perform exploratory data analysis:")
    print("1. Make sure you have Jupyter installed:")
    print("   pip install jupyter")
    print("\n2. Navigate to the notebooks directory:")
    print("   cd notebooks")
    print("\n3. Launch Jupyter Notebook:")
    print("   jupyter notebook eda_visualization.ipynb")
    print("\nThe notebook contains comprehensive data analysis including:")
    print("  - Data quality checks")
    print("  - Statistical analysis")
    print("  - Visualizations")
    print("  - Correlation analysis")
    print("  - Feature selection recommendations")
    print("=" * 60 + "\n")


def run_preprocessing(top_k=None, corr_threshold=None):
    """
    Run data preprocessing pipeline.
    """
    print("\n" + "=" * 60)
    print(" " * 15 + "RUNNING PREPROCESSING")
    print("=" * 60 + "\n")
    
    # Set random seeds
    set_random_seeds(42)
    
    # Setup logging
    logger = setup_logging('preprocessing.log')
    
    # Ensure directory structure exists
    ensure_directory_structure()
    
    # Run preprocessing
    preprocessor = DataPreprocessor(
        data_path='data/Variables_Horno.csv',
        train_ratio=0.7,
        top_k=top_k,
        corr_threshold=corr_threshold
    )
    
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Test samples: {X_test.shape[0]}")
    print(f"  - Number of features: {X_train.shape[1]}")
    print(f"  - Target variable: {preprocessor.target_name}")
    print(f"\nSaved:")
    print(f"  - Scaler: models/scaler.pkl")
    print("=" * 60 + "\n")
    
    return X_train, X_test, y_train, y_test, preprocessor


def run_training(top_k=None, corr_threshold=None):
    """
    Run model training pipeline.
    """
    if not _tensorflow_available:
        print("\n" + "=" * 60)
        print(" " * 15 + "TENSORFLOW NOT AVAILABLE")
        print("=" * 60 + "\n")
        print("TensorFlow is not installed or not compatible with your Python version.")
        print("Please install TensorFlow or use a compatible Python version (3.8-3.11).")
        print("\nTo install TensorFlow:")
        print("  pip install tensorflow")
        print("\nOr use Python 3.11:")
        print("  python3.11 -m venv venv")
        print("  source venv/bin/activate")
        print("  pip install -r requirements.txt")
        print("=" * 60 + "\n")
        return
    
    print("\n" + "=" * 60)
    print(" " * 15 + "RUNNING MODEL TRAINING")
    print("=" * 60 + "\n")
    
    # Set random seeds
    set_random_seeds(42)
    
    # Setup logging
    logger = setup_logging('training.log')
    
    # Ensure directory structure exists
    ensure_directory_structure()
    
    # Check if data file exists
    if not os.path.exists('data/Variables_Horno.csv'):
        print("ERROR: Data file 'data/Variables_Horno.csv' not found!")
        print("Please place the Variables_Horno.csv file in the 'data/' directory.")
        return
    
    # Preprocess data
    print("Step 1/3: Preprocessing data...")
    preprocessor = DataPreprocessor(data_path='data/Variables_Horno.csv', train_ratio=0.7, top_k=top_k, corr_threshold=corr_threshold)
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    
    # Determine task type
    task_type = determine_task_type(y_train)
    
    # Build model
    print("\nStep 2/3: Building CNN model...")
    input_shape = (X_train.shape[1],)
    cnn_model = CNNModel(input_shape=input_shape, task_type=task_type)
    cnn_model.build_model()
    cnn_model.compile_model()
    
    print("\nModel Architecture:")
    cnn_model.summary()
    
    # Train model
    print("\nStep 3/3: Training model...")
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
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nSaved:")
    print(f"  - Model: models/cnn_model.h5")
    print(f"  - Best model: models/cnn_model_best.h5")
    print(f"  - Scaler: models/scaler.pkl")
    print(f"  - Training curves: results/training_curves.png")
    print("\nNext step: Run evaluation with:")
    print("  python main.py --evaluate")
    print("=" * 60 + "\n")


def run_evaluation(baseline_path=None, top_k=None, corr_threshold=None):
    """
    Run model evaluation pipeline.
    """
    if not _tensorflow_available:
        print("\n" + "=" * 60)
        print(" " * 15 + "TENSORFLOW NOT AVAILABLE")
        print("=" * 60 + "\n")
        print("TensorFlow is not installed or not compatible with your Python version.")
        print("Please install TensorFlow or use a compatible Python version (3.8-3.11).")
        print("\nTo install TensorFlow:")
        print("  pip install tensorflow")
        print("\nOr use Python 3.11:")
        print("  python3.11 -m venv venv")
        print("  source venv/bin/activate")
        print("  pip install -r requirements.txt")
        print("=" * 60 + "\n")
        return
    
    print("\n" + "=" * 60)
    print(" " * 15 + "RUNNING MODEL EVALUATION")
    print("=" * 60 + "\n")
    # Generate PDF report
    report_path = generate_pdf_report(
        metrics_path='results/metrics.json',
        training_curves_path='results/training_curves.png',
        confusion_matrix_path='results/confusion_matrix.png',
        output_path='results/report.pdf'
    )
    print(f"Report generated: {report_path}")
    
    # Set random seeds
    set_random_seeds(42)
    
    # Setup logging
    logger = setup_logging('evaluation.log')
    
    # Ensure directory structure exists
    ensure_directory_structure()
    
    # Check if model exists
    if not os.path.exists('models/cnn_model.h5'):
        print("ERROR: Trained model not found!")
        print("Please run training first with:")
        print("  python main.py --train")
        return
    
    # Check if data file exists
    if not os.path.exists('data/Variables_Horno.csv'):
        print("ERROR: Data file 'data/Variables_Horno.csv' not found!")
        print("Please place the Variables_Horno.csv file in the 'data/' directory.")
        return
    
    # Load and preprocess data
    print("Step 1/3: Loading and preprocessing data...")
    preprocessor = DataPreprocessor(data_path='data/Variables_Horno.csv', train_ratio=0.7, top_k=top_k, corr_threshold=corr_threshold)
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    
    # Initialize evaluator
    print("\nStep 2/3: Loading trained model...")
    evaluator = ModelEvaluator(
        model_path='models/cnn_model.h5',
        scaler_path='models/scaler.pkl',
        baseline_path=baseline_path
    )
    evaluator.load_model_and_scaler()
    
    # Evaluate model
    print("\nStep 3/3: Evaluating model on test set...")
    metrics = evaluator.evaluate(X_test, y_test)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save metrics
    save_metrics_to_json(metrics, 'results/metrics.json')
    
    # Generate detailed report
    y_pred = evaluator.predict(X_test)
    evaluator.generate_detailed_report(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  - Metrics: results/metrics.json")
    print(f"  - Training curves: results/training_curves.png")
    if evaluator.task_type == 'classification':
        print(f"  - Confusion matrix: results/confusion_matrix.png")
    print("=" * 60 + "\n")


def main():
    """
    Main function to parse arguments and run appropriate pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Horno Prediction Project - CNN for POWER ON Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --eda          # Show EDA instructions
  python main.py --preprocess   # Run data preprocessing
  python main.py --train        # Train the CNN model
  python main.py --evaluate     # Evaluate the trained model
  python main.py --all          # Run complete pipeline (train + evaluate)
        """
    )
    
    parser.add_argument('--eda', action='store_true',
                        help='Show instructions for running EDA')
    parser.add_argument('--preprocess', action='store_true',
                        help='Run data preprocessing only')
    parser.add_argument('--train', action='store_true',
                        help='Train the CNN model')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate the trained model')
    parser.add_argument('--all', action='store_true',
                        help='Run complete pipeline (train + evaluate)')
    # Optional enhancements
    parser.add_argument('--top_k', type=int, default=None,
                        help='Select top-k features by absolute correlation (computed on train set)')
    parser.add_argument('--corr_threshold', type=float, default=None,
                        help='Select features with |correlation| >= threshold (computed on train set)')
    parser.add_argument('--baseline', type=str, default=None,
                        help='Path to baseline model pickle to compare during evaluation')
    parser.add_argument('--report', action='store_true',
                        help='Generate PDF report from existing results (results/metrics.json, images)')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Run appropriate pipeline
    if args.eda:
        run_eda()
    
    if args.preprocess:
        run_preprocessing(top_k=args.top_k, corr_threshold=args.corr_threshold)
    
    if args.train or args.all:
        run_training(top_k=args.top_k, corr_threshold=args.corr_threshold)
    
    if args.evaluate or args.all:
        run_evaluation(baseline_path=args.baseline, top_k=args.top_k, corr_threshold=args.corr_threshold)

    if args.report:
        path = generate_pdf_report(
            metrics_path='results/metrics.json',
            training_curves_path='results/training_curves.png',
            confusion_matrix_path='results/confusion_matrix.png',
            output_path='results/report.pdf'
        )
        print(f"Report generated: {path}")


if __name__ == '__main__':
    main()

