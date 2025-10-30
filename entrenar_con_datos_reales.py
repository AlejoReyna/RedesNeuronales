#!/usr/bin/env python3
"""
Script para Entrenar Modelo con Datos Reales del Horno
=======================================================
Este script entrena el modelo CNN usando el archivo REAL de variables del horno.

Uso:
    python3 entrenar_con_datos_reales.py
    
El modelo se detectará automáticamente como REGRESIÓN (no clasificación)
porque el target 'P. On (min)' tiene valores continuos.
"""

import os
import sys
import logging
from pathlib import Path

# Agregar src al path
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
    """Función principal para entrenar con datos reales"""
    
    print("=" * 80)
    print("ENTRENAMIENTO CON DATOS REALES DEL HORNO")
    print("=" * 80)
    print()
    
    # Configurar semillas aleatorias
    set_random_seeds(42)
    
    # Setup logging
    setup_logging('training_real_data.log')
    
    # Asegurar estructura de directorios
    ensure_directory_structure()
    
    # Ruta al archivo real
    data_path = 'data/Variables_Horno_REAL.csv'
    
    if not os.path.exists(data_path):
        print(f"ERROR: No se encuentra el archivo {data_path}")
        print("Por favor, asegúrate de que el archivo esté en el directorio data/")
        return
    
    print(f"✓ Archivo encontrado: {data_path}")
    print()
    
    # Paso 1: Preprocesamiento
    print("=" * 80)
    print("PASO 1: PREPROCESAMIENTO DE DATOS")
    print("=" * 80)
    print()
    
    preprocessor = DataPreprocessor(
        data_path=data_path,
        train_ratio=0.7
    )
    
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    
    print()
    print("Resumen del preprocesamiento:")
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Test samples: {X_test.shape[0]}")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"  - Target: {preprocessor.target_name}")
    print(f"  - Target range: [{y_train.min():.2f}, {y_train.max():.2f}] minutos")
    print()
    
    # Paso 2: Determinar tipo de tarea
    print("=" * 80)
    print("PASO 2: DETECCIÓN DEL TIPO DE TAREA")
    print("=" * 80)
    print()
    
    task_type = determine_task_type(y_train)
    
    if task_type == 'regression':
        print("✓ Tipo detectado: REGRESIÓN")
        print("  El modelo predecirá valores continuos (tiempo en minutos)")
        print(f"  Rango del target: {y_train.min():.2f} - {y_train.max():.2f} minutos")
    else:
        print("✓ Tipo detectado: CLASIFICACIÓN")
    
    print()
    
    # Paso 3: Construir modelo
    print("=" * 80)
    print("PASO 3: CONSTRUCCIÓN DEL MODELO CNN")
    print("=" * 80)
    print()
    
    input_shape = (X_train.shape[1],)
    cnn_model = CNNModel(input_shape=input_shape, task_type=task_type)
    cnn_model.build_model()
    cnn_model.compile_model()
    
    print("\nArquitectura del modelo:")
    print("-" * 80)
    cnn_model.summary()
    print()
    
    # Paso 4: Entrenar modelo
    print("=" * 80)
    print("PASO 4: ENTRENAMIENTO DEL MODELO")
    print("=" * 80)
    print()
    
    print("Entrenando modelo...")
    print("(Esto puede tomar varios minutos)")
    print()
    
    history = cnn_model.train(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.15
    )
    
    # Guardar modelo
    model_path = 'models/cnn_model_real.h5'
    cnn_model.save_model(model_path)
    
    # Guardar scaler de features y target
    scaler_path = 'models/scaler.pkl'
    preprocessor.save_scaler(scaler_path)
    if task_type == 'regression':
        cnn_model.save_target_scaler('models/target_scaler.pkl')
    
    # Guardar curvas de entrenamiento
    save_training_curves(history, 'results/training_curves_real.png')
    
    print()
    print("✓ Modelo entrenado y guardado")
    print(f"  - Modelo: {model_path}")
    print(f"  - Scaler: {scaler_path}")
    print(f"  - Curvas: results/training_curves_real.png")
    print()
    
    # Paso 5: Evaluación
    print("=" * 80)
    print("PASO 5: EVALUACIÓN DEL MODELO")
    print("=" * 80)
    print()
    
    # Los conjuntos ya están escalados tras preprocess(); conservar copia explícita
    X_test_scaled = X_test.copy()
    
    evaluator = ModelEvaluator(
        model_path=model_path,
        scaler_path=scaler_path,
        target_scaler_path='models/target_scaler.pkl'
    )
    evaluator.load_model_and_scaler()
    
    # Usar los datos ya escalados con el scaler correcto
    metrics = evaluator.evaluate(X_test, y_test, X_test_scaled=X_test_scaled)
    
    # Guardar métricas
    metrics_path = 'results/metrics_real.json'
    save_metrics_to_json(metrics, metrics_path)
    
    # Mostrar métricas
    print()
    print("Métricas de evaluación:")
    print("-" * 80)
    print_metrics(metrics)
    print()
    
    # Generar reporte detallado
    y_pred = evaluator.predict(X_test_scaled, already_scaled=True)
    evaluator.generate_detailed_report(y_test, y_pred)
    
    # Análisis adicional para regresión
    if task_type == 'regression':
        print()
        print("=" * 80)
        print("ANÁLISIS DE REGRESIÓN")
        print("=" * 80)
        print()
        
        import numpy as np
        
        mae = metrics.get('mae', 0)
        rmse = metrics.get('rmse', 0)
        r2 = metrics.get('r2_score', 0)
        
        print(f"MAE (Mean Absolute Error): {mae:.2f} minutos")
        print(f"RMSE (Root Mean Squared Error): {rmse:.2f} minutos")
        print(f"R² (Coeficiente de determinación): {r2:.4f}")
        print()
        
        # Interpretación
        print("Interpretación:")
        print("-" * 80)
        if r2 >= 0.9:
            print("✓ Excelente: R² ≥ 0.9 - El modelo explica más del 90% de la varianza")
        elif r2 >= 0.8:
            print("✓ Muy bueno: R² ≥ 0.8 - El modelo explica más del 80% de la varianza")
        elif r2 >= 0.7:
            print("✓ Bueno: R² ≥ 0.7 - El modelo explica más del 70% de la varianza")
        elif r2 >= 0.5:
            print("⚠ Moderado: R² ≥ 0.5 - El modelo explica más del 50% de la varianza")
        else:
            print("⚠ Bajo: R² < 0.5 - El modelo necesita mejoras")
        
        print()
        print(f"El error promedio (MAE) es de {mae:.2f} minutos.")
        print(f"El error típico (RMSE) es de {rmse:.2f} minutos.")
        
        # Comparar con el target
        target_mean = y_test.mean()
        mae_percent = (mae / target_mean) * 100 if target_mean > 0 else 0
        print(f"El error promedio representa aproximadamente {mae_percent:.1f}% del valor promedio del target.")
        print()
    
    print("=" * 80)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 80)
    print()
    print("Archivos generados:")
    print(f"  - Modelo: {model_path}")
    print(f"  - Scaler: {scaler_path}")
    print(f"  - Métricas: {metrics_path}")
    print(f"  - Curvas de entrenamiento: results/training_curves_real.png")
    print()
    print("Para evaluar nuevamente:")
    print(f"  python3 -c \"from src.evaluate import ModelEvaluator; eval = ModelEvaluator('{model_path}', '{scaler_path}'); eval.load_model_and_scaler(); from src.preprocess import DataPreprocessor; p = DataPreprocessor('{data_path}'); X_train, X_test, y_train, y_test = p.preprocess(); metrics = eval.evaluate(X_test, y_test); print(metrics)\"")
    print()


if __name__ == '__main__':
    main()

