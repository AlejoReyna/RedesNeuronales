"""Evaluate an ensemble of trained CNN models by averaging their predictions."""

from pathlib import Path
from typing import List

import numpy as np

from src.preprocess import DataPreprocessor
from src.evaluate import ModelEvaluator
from src.utils import set_random_seeds, save_metrics_to_json


def load_models(model_dirs: List[Path]) -> List[ModelEvaluator]:
    evaluators = []
    for directory in model_dirs:
        model_path = directory / "cnn_model.h5"
        scaler_path = directory / "scaler.pkl"
        target_scaler_path = directory / "target_scaler.pkl"

        if not model_path.exists():
            continue

        evaluator = ModelEvaluator(
            model_path=str(model_path),
            scaler_path=str(scaler_path) if scaler_path.exists() else "models/scaler.pkl",
            target_scaler_path=str(target_scaler_path) if target_scaler_path.exists() else "models/target_scaler.pkl",
        )
        evaluator.load_model_and_scaler()
        evaluators.append(evaluator)

    return evaluators


def evaluate_ensemble(model_dirs: List[Path], data_path: str = "data/Variables_Horno_REAL.csv") -> None:
    if not model_dirs:
        print("No ensemble models found.")
        return

    set_random_seeds(42)

    preprocessor = DataPreprocessor(data_path=data_path, train_ratio=0.7)
    X_train, X_test, y_train, y_test = preprocessor.preprocess(fit_scaler=False)

    evaluators = load_models(model_dirs)
    if not evaluators:
        print("Could not load any ensemble models.")
        return

    ensemble_predictions = []
    for evaluator in evaluators:
        evaluator.detect_task_type(y_test)
        preds = evaluator.predict(X_test, already_scaled=True)
        ensemble_predictions.append(preds)

    stacked_preds = np.stack(ensemble_predictions, axis=0)
    averaged_preds = stacked_preds.mean(axis=0)

    metrics = evaluators[0].calculate_regression_metrics(y_test, averaged_preds)
    metrics['ensemble_size'] = len(ensemble_predictions)

    print("\nEnsemble Metrics:\n" + "-" * 40)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    save_metrics_to_json(metrics, "results/ensemble_metrics.json")



def main() -> None:
    ensemble_dir = Path("models/ensemble")
    if not ensemble_dir.exists():
        print("Ensemble directory not found. Train models with train_ensemble.py first.")
        return

    model_dirs = sorted([p for p in ensemble_dir.iterdir() if p.is_dir()])
    evaluate_ensemble(model_dirs)


if __name__ == "__main__":
    main()


