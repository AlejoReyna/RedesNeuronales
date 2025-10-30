"""Train a small ensemble of CNN models with different random seeds."""

import os
import shutil
from pathlib import Path

from src.preprocess import DataPreprocessor
from src.train_cnn import CNNModel, determine_task_type
from src.utils import set_random_seeds, save_training_curves, ensure_directory_structure


def train_single_model(seed: int, data_path: str, output_dir: Path) -> None:
    """Train a single CNN instance with a given random seed."""

    set_random_seeds(seed)

    preprocessor = DataPreprocessor(data_path=data_path, train_ratio=0.7)
    X_train, X_test, y_train, y_test = preprocessor.preprocess()

    task_type = determine_task_type(y_train)
    model = CNNModel(input_shape=(X_train.shape[1],), task_type=task_type)
    model.build_model()
    model.compile_model()

    history = model.train(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.15,
    )

    # Persist artifacts per seed
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    model_path = seed_dir / "cnn_model.h5"
    model.save_model(str(model_path))

    target_scaler_path = seed_dir / "target_scaler.pkl"
    model.save_target_scaler(str(target_scaler_path))

    # Copy feature scaler to ensure reproducibility
    global_scaler = Path("models/scaler.pkl")
    if global_scaler.exists():
        shutil.copy(global_scaler, seed_dir / "scaler.pkl")

    # Save training curves for quick inspection
    curves_path = seed_dir / "training_curves.png"
    save_training_curves(history, str(curves_path))


def main() -> None:
    ensure_directory_structure()

    data_path = "data/Variables_Horno_REAL.csv"
    seeds = [0, 1, 2]
    output_dir = Path("models/ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        print("=" * 80)
        print(f"Training ensemble member with seed {seed}")
        print("=" * 80)
        train_single_model(seed, data_path, output_dir)

    print("\nEnsemble training completed. Models saved under models/ensemble/")


if __name__ == "__main__":
    main()


