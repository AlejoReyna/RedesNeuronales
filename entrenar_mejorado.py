#!/usr/bin/env python3
"""
Script Mejorado para Entrenar Modelo con Datos Reales
======================================================
Corrige el problema de escalado del target y mejora el entrenamiento.

Mejoras:
1. Escala el target (CRÍTICO para redes neuronales)
2. Ajusta hiperparámetros para mejor convergencia
3. Mejor manejo de early stopping
"""

import os
import sys
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import DataPreprocessor
from src.train_cnn import CNNModel, determine_task_type
from src.evaluate import ModelEvaluator
from src.utils import (
    setup_logging, save_training_curves, set_random_seeds,
    ensure_directory_structure, print_metrics, save_metrics_to_json
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Función principal mejorada"""
    
    print("=" * 80)
    print("ENTRENAMIENTO MEJORADO CON DATOS REALES")
    print("=" * 80)
    print()
    print("MEJORAS APLICADAS:")
    print("  1. Escalado del target (CRÍTICO)")
    print("  2. Hiperparámetros ajustados")
    print("  3. Mejor manejo de early stopping")
    print()
    
    # Configurar semillas
    set_random_seeds(42)
    setup_logging('training_improved.log')
    ensure_directory_structure()
    
    # Cargar datos
    data_path = 'data/Variables_Horno_REAL.csv'
    preprocessor = DataPreprocessor(data_path=data_path, train_ratio=0.7)
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    
    print(f"✓ Datos cargados: {len(X_train)} train, {len(X_test)} test")
    print(f"✓ Target rango: {y_train.min():.2f} - {y_train.max():.2f} minutos")
    print()
    
    # MEJORA CRÍTICA: Escalar el target también
    print("=" * 80)
    print("ESCALANDO TARGET (MEJORA CRÍTICA)")
    print("=" * 80)
    print()
    
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"Target original - Media: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
    print(f"Target escalado  - Media: {y_train_scaled.mean():.4f}, Std: {y_train_scaled.std():.4f}")
    print()
    
    # Guardar target scaler
    target_scaler_path = 'models/target_scaler.pkl'
    os.makedirs('models', exist_ok=True)
    joblib.dump(target_scaler, target_scaler_path)
    print(f"✓ Target scaler guardado en: {target_scaler_path}")
    print()
    
    # Determinar tipo de tarea
    task_type = determine_task_type(y_train_scaled)
    print(f"✓ Tipo de tarea: {task_type}")
    print()
    
    # Construir modelo
    print("=" * 80)
    print("CONSTRUYENDO MODELO")
    print("=" * 80)
    print()
    
    input_shape = (X_train.shape[1],)
    cnn_model = CNNModel(input_shape=input_shape, task_type=task_type)
    cnn_model.build_model()
    
    # Compilar con learning rate más bajo para mejor convergencia
    cnn_model.model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Más bajo que antes
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    print("✓ Modelo construido y compilado")
    print()
    
    # Entrenar con más paciencia
    print("=" * 80)
    print("ENTRENANDO MODELO")
    print("=" * 80)
    print()
    
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,  # Más paciencia
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'models/cnn_model_improved_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history = cnn_model.model.fit(
        X_train, y_train_scaled,
        epochs=200,  # Más épocas
        batch_size=32,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1
    )
    
    # Guardar modelo
    model_path = 'models/cnn_model_improved.h5'
    cnn_model.model.save(model_path)
    print(f"✓ Modelo guardado en: {model_path}")
    
    # Guardar scaler de features
    scaler_path = 'models/scaler_improved.pkl'
    preprocessor.save_scaler(scaler_path)
    
    # Guardar curvas
    save_training_curves(history, 'results/training_curves_improved.png')
    print()
    
    # Evaluar
    print("=" * 80)
    print("EVALUANDO MODELO")
    print("=" * 80)
    print()
    
    # Predecir en escala
    X_test_scaled = preprocessor.scaler.transform(X_test)
    y_pred_scaled = cnn_model.model.predict(X_test_scaled, verbose=0)
    
    # Desescalar predicciones
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calcular métricas en escala original
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("MÉTRICAS EN ESCALA ORIGINAL (minutos):")
    print("-" * 80)
    print(f"MAE:  {mae:.2f} minutos")
    print(f"RMSE: {rmse:.2f} minutos")
    print(f"R²:   {r2:.4f}")
    print()
    
    print("ANÁLISIS:")
    print("-" * 80)
    print(f"Target promedio: {y_test.mean():.2f} minutos")
    print(f"Error promedio: {mae:.2f} minutos ({mae/y_test.mean()*100:.1f}% del promedio)")
    print()
    
    if r2 >= 0.7:
        print("✓ EXCELENTE: R² ≥ 0.7")
    elif r2 >= 0.5:
        print("✓ BUENO: R² ≥ 0.5")
    elif r2 >= 0.3:
        print("⚠ MODERADO: R² ≥ 0.3 - Mejorable")
    elif r2 >= 0.0:
        print("⚠ BAJO: R² > 0 pero necesita mejoras")
    else:
        print("⚠ MUY BAJO: R² < 0 - Requiere revisión")
    
    print()
    print("=" * 80)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 80)
    print()
    print("Archivos generados:")
    print(f"  - Modelo: {model_path}")
    print(f"  - Feature scaler: {scaler_path}")
    print(f"  - Target scaler: {target_scaler_path}")
    print(f"  - Curvas: results/training_curves_improved.png")
    print()


if __name__ == '__main__':
    # Importar keras aquí para evitar problemas de importación
    import tensorflow as tf
    from tensorflow import keras
    main()

