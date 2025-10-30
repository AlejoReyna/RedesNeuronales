#!/usr/bin/env python3
"""
Evaluación del Modelo con Cross-Validation Estratificada
==========================================================
Este script evalúa el modelo usando cross-validation estratificada de 5 folds,
proporcionando métricas más robustas y creíbles que un simple train-test split.

Uso:
    python3 evaluar_con_cross_validation.py
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import os
import sys
from datetime import datetime

# Agregar src al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train_cnn import CNNModel, determine_task_type
from src.utils import set_random_seeds

def run_cross_validation_evaluation(n_splits=5, epochs=50, verbose=True):
    """
    Ejecuta cross-validation estratificada para evaluación robusta.
    
    Args:
        n_splits (int): Número de folds (default: 5)
        epochs (int): Número de épocas por fold (default: 50)
        verbose (bool): Mostrar progreso detallado
    
    Returns:
        dict: Resultados de cross-validation
    """
    print("=" * 80)
    print("EVALUACIÓN CON CROSS-VALIDATION ESTRATIFICADA")
    print("=" * 80)
    print()
    print(f"Configuración:")
    print(f"  - Folds: {n_splits}")
    print(f"  - Épocas por fold: {epochs}")
    print(f"  - Random seed: 42")
    print()
    
    # Cargar datos
    data_path = 'data/Variables_Horno.csv'
    if not os.path.exists(data_path):
        print(f"ERROR: No se encuentra {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    feature_names = df.columns[1:].tolist()
    
    print(f"Datos cargados: {len(X)} muestras, {len(feature_names)} features")
    print(f"Distribución de clases: Clase 0 = {(y==0).sum()}, Clase 1 = {(y==1).sum()}")
    print()
    
    # Configurar cross-validation estratificada
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Métricas a recolectar
    fold_results = []
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    
    # Entrenar y evaluar en cada fold
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
        print("-" * 80)
        print(f"FOLD {fold}/{n_splits}")
        print("-" * 80)
        
        # Split de datos
        X_train_fold = X[train_idx]
        X_test_fold = X[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]
        
        print(f"Train: {len(X_train_fold)} muestras (Clase 0: {(y_train_fold==0).sum()}, Clase 1: {(y_train_fold==1).sum()})")
        print(f"Test:  {len(X_test_fold)} muestras (Clase 0: {(y_test_fold==0).sum()}, Clase 1: {(y_test_fold==1).sum()})")
        
        # Escalar datos (ajustar solo con train)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)
        
        # Determinar tipo de tarea
        task_type = determine_task_type(y_train_fold)
        
        # Construir modelo
        input_shape = (X_train_scaled.shape[1],)
        model = CNNModel(input_shape=input_shape, task_type=task_type)
        model.build_model()
        model.compile_model()
        
        # Entrenar modelo
        if verbose:
            print(f"Entrenando modelo (épocas: {epochs})...")
        
        history = model.train(
            X_train_scaled, y_train_fold,
            epochs=epochs,
            batch_size=32,
            validation_split=0.15
        )
        
        # Predecir
        y_pred_proba = model.predict(X_test_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calcular métricas
        acc = accuracy_score(y_test_fold, y_pred)
        prec = precision_score(y_test_fold, y_pred, zero_division=0)
        rec = recall_score(y_test_fold, y_pred, zero_division=0)
        f1 = f1_score(y_test_fold, y_pred, zero_division=0)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test_fold, y_pred)
        
        # Guardar resultados del fold
        fold_result = {
            'fold': fold,
            'train_samples': len(X_train_fold),
            'test_samples': len(X_test_fold),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        }
        fold_results.append(fold_result)
        
        all_accuracies.append(acc)
        all_precisions.append(prec)
        all_recalls.append(rec)
        all_f1_scores.append(f1)
        
        if verbose:
            print(f"  ✓ Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
            print(f"  ✓ Precision: {prec:.4f}")
            print(f"  ✓ Recall:    {rec:.4f}")
            print(f"  ✓ F1-Score:  {f1:.4f}")
            print(f"  ✓ Confusion Matrix:")
            print(f"        Predicted")
            print(f"        0    1")
            print(f"   0   {cm[0,0]:4d} {cm[0,1]:4d}")
            print(f"   1   {cm[1,0]:4d} {cm[1,1]:4d}")
        
        print()
    
    # Calcular estadísticas agregadas
    results = {
        'n_splits': n_splits,
        'epochs_per_fold': epochs,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'overall_stats': {
            'accuracy': {
                'mean': float(np.mean(all_accuracies)),
                'std': float(np.std(all_accuracies)),
                'min': float(np.min(all_accuracies)),
                'max': float(np.max(all_accuracies))
            },
            'precision': {
                'mean': float(np.mean(all_precisions)),
                'std': float(np.std(all_precisions)),
                'min': float(np.min(all_precisions)),
                'max': float(np.max(all_precisions))
            },
            'recall': {
                'mean': float(np.mean(all_recalls)),
                'std': float(np.std(all_recalls)),
                'min': float(np.min(all_recalls)),
                'max': float(np.max(all_recalls))
            },
            'f1_score': {
                'mean': float(np.mean(all_f1_scores)),
                'std': float(np.std(all_f1_scores)),
                'min': float(np.min(all_f1_scores)),
                'max': float(np.max(all_f1_scores))
            }
        },
        'fold_results': fold_results
    }
    
    # Mostrar resumen
    print("=" * 80)
    print("RESULTADOS DE CROSS-VALIDATION")
    print("=" * 80)
    print()
    print("Métricas agregadas (promedio ± desviación estándar):")
    print("-" * 80)
    print(f"Accuracy:  {results['overall_stats']['accuracy']['mean']:.4f} ± {results['overall_stats']['accuracy']['std']:.4f}")
    print(f"          Rango: [{results['overall_stats']['accuracy']['min']:.4f}, {results['overall_stats']['accuracy']['max']:.4f}]")
    print()
    print(f"Precision: {results['overall_stats']['precision']['mean']:.4f} ± {results['overall_stats']['precision']['std']:.4f}")
    print(f"          Rango: [{results['overall_stats']['precision']['min']:.4f}, {results['overall_stats']['precision']['max']:.4f}]")
    print()
    print(f"Recall:    {results['overall_stats']['recall']['mean']:.4f} ± {results['overall_stats']['recall']['std']:.4f}")
    print(f"          Rango: [{results['overall_stats']['recall']['min']:.4f}, {results['overall_stats']['recall']['max']:.4f}]")
    print()
    print(f"F1-Score:  {results['overall_stats']['f1_score']['mean']:.4f} ± {results['overall_stats']['f1_score']['std']:.4f}")
    print(f"          Rango: [{results['overall_stats']['f1_score']['min']:.4f}, {results['overall_stats']['f1_score']['max']:.4f}]")
    print()
    
    # Análisis de consistencia
    print("Análisis de consistencia entre folds:")
    print("-" * 80)
    cv_std = results['overall_stats']['accuracy']['std']
    if cv_std < 0.01:
        print("✓ Muy consistente: Desviación estándar < 0.01")
        print("  El modelo mantiene desempeño similar en todos los folds")
    elif cv_std < 0.05:
        print("✓ Consistente: Desviación estándar < 0.05")
        print("  El modelo muestra variabilidad moderada entre folds")
    else:
        print("⚠ Variabilidad: Desviación estándar ≥ 0.05")
        print("  El modelo muestra variabilidad significativa entre folds")
        print("  Esto puede indicar dependencia de la partición específica")
    print()
    
    # Guardar resultados
    os.makedirs('results', exist_ok=True)
    results_path = 'results/cross_validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Resultados guardados en: {results_path}")
    print()
    
    return results

def main():
    """Función principal"""
    # Configurar semilla aleatoria
    set_random_seeds(42)
    
    # Ejecutar cross-validation
    results = run_cross_validation_evaluation(n_splits=5, epochs=50, verbose=True)
    
    if results:
        print("=" * 80)
        print("EVALUACIÓN COMPLETADA")
        print("=" * 80)
        print()
        print("Interpretación de resultados:")
        print("-" * 80)
        mean_acc = results['overall_stats']['accuracy']['mean']
        std_acc = results['overall_stats']['accuracy']['std']
        
        if mean_acc >= 0.99:
            print("⚠ El modelo mantiene precisión muy alta (≥99%) en cross-validation.")
            print("  Esto confirma que el problema es altamente separable con estas features.")
            print("  Sin embargo, el 100% es específico del split temporal.")
        elif mean_acc >= 0.95:
            print("✓ El modelo tiene buena precisión (≥95%) en cross-validation.")
            print("  Esto es más realista que el 100% reportado con split temporal.")
        else:
            print("✓ El modelo muestra métricas más realistas en cross-validation.")
            print("  El split temporal puede haber favorecido un desempeño mejor.")
        
        print()
        print("Recomendación:")
        print("  Usar estos resultados de cross-validation como métricas principales")
        print("  en lugar del 100% obtenido con el split temporal.")
        print()

if __name__ == "__main__":
    main()

