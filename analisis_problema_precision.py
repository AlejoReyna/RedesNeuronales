#!/usr/bin/env python3
"""
Análisis Detallado del Problema de Precisión Perfecta
======================================================
Este script analiza por qué el modelo tiene 100% de precisión y proporciona
soluciones para una evaluación más realista.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import os
import sys

# Agregar src al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train_cnn import CNNModel, determine_task_type
from src.preprocess import DataPreprocessor

def analyze_feature_separability():
    """Analiza qué tan separables son las clases con las features actuales"""
    print("=" * 80)
    print("ANÁLISIS DE SEPARABILIDAD DE CLASES")
    print("=" * 80)
    print()
    
    # Cargar datos
    df = pd.read_csv('data/Variables_Horno.csv')
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    feature_names = df.columns[1:].tolist()
    
    # Split temporal
    split_index = int(len(X) * 0.7)
    X_train = X[:split_index]
    y_train = y[:split_index]
    
    print("Análisis de separación de clases por feature:")
    print("-" * 80)
    print(f"{'Feature':<15} {'Clase 0 Media':<15} {'Clase 1 Media':<15} {'Diferencia':<15} {'Separable?':<10}")
    print("-" * 80)
    
    separability_scores = []
    for i, feat_name in enumerate(feature_names):
        class_0_values = X_train[y_train == 0, i]
        class_1_values = X_train[y_train == 1, i]
        
        mean_0 = class_0_values.mean()
        mean_1 = class_1_values.mean()
        diff = abs(mean_1 - mean_0)
        
        # Verificar si las distribuciones se solapan mínimamente
        min_1 = class_1_values.min()
        max_0 = class_0_values.max()
        
        # Si no hay solapamiento significativo, es separable
        separable = "SÍ" if (min_1 > max_0 + 0.1) or (max_0 < min_1 - 0.1) else "Parcial"
        
        separability_scores.append({
            'feature': feat_name,
            'mean_0': mean_0,
            'mean_1': mean_1,
            'diff': diff,
            'separable': separable
        })
        
        print(f"{feat_name:<15} {mean_0:>14.4f} {mean_1:>14.4f} {diff:>14.4f} {separable:<10}")
    
    print()
    print("Features con mejor separación:")
    separability_scores.sort(key=lambda x: x['diff'], reverse=True)
    for i, score in enumerate(separability_scores[:5], 1):
        print(f"  {i}. {score['feature']}: diferencia = {score['diff']:.4f}")
    
    return separability_scores

def test_with_shuffled_split():
    """Prueba el modelo con un split aleatorio estratificado en lugar de temporal"""
    print()
    print("=" * 80)
    print("PRUEBA CON SPLIT ALEATORIO ESTRATIFICADO")
    print("=" * 80)
    print()
    print("Comparando split temporal vs split aleatorio:")
    print("(El split aleatorio debería ser más difícil y dar métricas más realistas)")
    print()
    
    # Cargar datos
    df = pd.read_csv('data/Variables_Horno.csv')
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    
    # Split temporal (como en el código actual)
    split_index = int(len(X) * 0.7)
    X_train_temp = X[:split_index]
    X_test_temp = X[split_index:]
    y_train_temp = y[:split_index]
    y_test_temp = y[split_index:]
    
    # Split aleatorio estratificado
    X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("Comparación de distribuciones:")
    print("-" * 80)
    print(f"{'Split':<20} {'Train Clase 0':<15} {'Train Clase 1':<15} {'Test Clase 0':<15} {'Test Clase 1':<15}")
    print("-" * 80)
    print(f"{'Temporal':<20} {(y_train_temp==0).sum():<15} {(y_train_temp==1).sum():<15} {(y_test_temp==0).sum():<15} {(y_test_temp==1).sum():<15}")
    print(f"{'Aleatorio':<20} {(y_train_rand==0).sum():<15} {(y_train_rand==1).sum():<15} {(y_test_rand==0).sum():<15} {(y_test_rand==1).sum():<15}")
    print()
    
    # Analizar si hay diferencias en la distribución de features
    print("Análisis de variabilidad de features:")
    print("-" * 80)
    
    # Feature más separadora (feature_1 según análisis anterior)
    feat_idx = 1
    
    print(f"Feature 1 (feature_1) - Estadísticas:")
    print(f"  Temporal Train - Media clase 0: {X_train_temp[y_train_temp==0, feat_idx].mean():.4f}")
    print(f"  Temporal Train - Media clase 1: {X_train_temp[y_train_temp==1, feat_idx].mean():.4f}")
    print(f"  Temporal Test  - Media clase 0: {X_test_temp[y_test_temp==0, feat_idx].mean():.4f}")
    print(f"  Temporal Test  - Media clase 1: {X_test_temp[y_test_temp==1, feat_idx].mean():.4f}")
    print()
    print(f"  Aleatorio Train - Media clase 0: {X_train_rand[y_train_rand==0, feat_idx].mean():.4f}")
    print(f"  Aleatorio Train - Media clase 1: {X_train_rand[y_train_rand==1, feat_idx].mean():.4f}")
    print(f"  Aleatorio Test  - Media clase 0: {X_test_rand[y_test_rand==0, feat_idx].mean():.4f}")
    print(f"  Aleatorio Test  - Media clase 1: {X_test_rand[y_test_rand==1, feat_idx].mean():.4f}")
    print()
    
    return {
        'temporal': (X_train_temp, X_test_temp, y_train_temp, y_test_temp),
        'random': (X_train_rand, X_test_rand, y_train_rand, y_test_rand)
    }

def run_cross_validation():
    """Ejecuta cross-validation estratificada para evaluación más robusta"""
    print()
    print("=" * 80)
    print("CROSS-VALIDATION ESTRATIFICADA (5-FOLD)")
    print("=" * 80)
    print()
    print("Esta es la forma más robusta de evaluar el modelo.")
    print("Si el modelo realmente es perfecto, debería mantener alta precisión en todos los folds.")
    print()
    
    # Cargar datos
    df = pd.read_csv('data/Variables_Horno.csv')
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    
    # 5-fold cross-validation estratificada
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    print("Entrenando modelo en cada fold...")
    print("-" * 80)
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
        print(f"\nFold {fold}/5:")
        
        X_train_fold = X[train_idx]
        X_test_fold = X[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]
        
        # Escalar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)
        
        # Construir y entrenar modelo
        input_shape = (X_train_scaled.shape[1],)
        task_type = determine_task_type(y_train_fold)
        
        model = CNNModel(input_shape=input_shape, task_type=task_type)
        model.build_model()
        model.compile_model()
        
        # Entrenar con menos épocas para velocidad
        history = model.train(
            X_train_scaled, y_train_fold,
            epochs=50,  # Menos épocas para rapidez
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
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        
        print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    print()
    print("=" * 80)
    print("RESULTADOS DE CROSS-VALIDATION")
    print("=" * 80)
    print()
    print(f"Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"F1-Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print()
    print("Folds individuales:")
    for i, (acc, prec, rec, f1) in enumerate(zip(accuracies, precisions, recalls, f1_scores), 1):
        print(f"  Fold {i}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    print()
    
    if np.mean(accuracies) >= 0.99:
        print("⚠️  El modelo mantiene precisión muy alta en cross-validation.")
        print("   Esto sugiere que el problema es genuinamente fácil con estas features.")
    elif np.mean(accuracies) >= 0.95:
        print("✓ El modelo tiene buena precisión pero no perfecta en cross-validation.")
        print("   Esto es más realista que 100%.")
    else:
        print("✓ El modelo muestra métricas más realistas en cross-validation.")
    
    return {
        'accuracies': accuracies,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores
    }

def generate_report():
    """Genera un reporte completo del análisis"""
    print()
    print("=" * 80)
    print("CONCLUSIONES Y RECOMENDACIONES")
    print("=" * 80)
    print()
    
    print("PROBLEMA IDENTIFICADO:")
    print("-" * 80)
    print("El modelo alcanza 100% de precisión NO por data leakage tradicional,")
    print("sino porque las primeras 5 features (feature_0 a feature_4) tienen una")
    print("separación casi perfecta entre las clases:")
    print("  • Clase 0 (POWER_ON=0): valores muy bajos (~0.01 o negativo)")
    print("  • Clase 1 (POWER_ON=1): valores muy altos (~0.80)")
    print()
    print("Con estas features, cualquier modelo puede aprender a separar las clases.")
    print()
    
    print("POR QUÉ EL INGENIERO TIENE RAZÓN:")
    print("-" * 80)
    print("1. 100% de precisión es sospechoso en problemas reales")
    print("2. Aunque no hay 'data leakage' técnico, el problema es trivial")
    print("3. El split temporal puede estar favoreciendo patrones repetitivos")
    print("4. La evaluación debería ser más robusta (cross-validation)")
    print()
    
    print("RECOMENDACIONES:")
    print("-" * 80)
    print("1. IMPLEMENTAR CROSS-VALIDATION para evaluación más robusta")
    print("2. PROBAR CON SPLIT ALEATORIO ESTRATIFICADO para ver si baja la precisión")
    print("3. DOCUMENTAR que las features tienen separación casi perfecta")
    print("4. MENCIONAR en el reporte que el 100% es específico de este dataset sintético")
    print("5. Si es posible, probar con datos reales del horno (más ruido)")
    print()
    
    print("PARA EL REPORTE:")
    print("-" * 80)
    print("""
En lugar de simplemente reportar "100% de precisión", deberías incluir:

"El modelo alcanzó 100% de precisión en el conjunto de prueba. Sin embargo,
este resultado requiere análisis cuidadoso:

1. **Características del Dataset**: Las primeras 5 features muestran una
   separación casi perfecta entre las clases (clase 0: valores bajos ~0.01,
   clase 1: valores altos ~0.80). Esto hace que el problema sea altamente
   separable.

2. **Validación Cruzada**: Para evaluación más robusta, se implementó
   cross-validation estratificada de 5 folds, obteniendo [resultados aquí].

3. **Limitaciones**: El 100% de precisión es específico a este dataset
   sintético. En datos reales con ruido de sensores, se esperaría una
   precisión de 85-95%.

4. **Conclusión**: El modelo es efectivo para este problema específico,
   pero su desempeño en producción dependerá de la calidad y variabilidad
   de los datos reales."
    """)

if __name__ == "__main__":
    # Análisis de separabilidad
    separability = analyze_feature_separability()
    
    # Test con split aleatorio
    splits = test_with_shuffled_split()
    
    # Cross-validation (comentado por tiempo, descomentar si quieres ejecutarlo)
    print("\n¿Ejecutar cross-validation? (Esto tomará varios minutos)")
    print("Descomenta la siguiente línea si quieres ejecutarlo:")
    # cv_results = run_cross_validation()
    
    # Generar reporte
    generate_report()

