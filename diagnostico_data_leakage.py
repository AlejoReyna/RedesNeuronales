#!/usr/bin/env python3
"""
Script de Diagnóstico de Data Leakage
======================================
Este script verifica si hay problemas de data leakage que puedan explicar
el 100% de precisión del modelo.

Verifica:
1. Duplicados entre train y test
2. Si el scaler se ajusta correctamente (solo con train)
3. Muestras idénticas entre train y test
4. Comparación de distribución de datos
5. Si hay correlaciones perfectas que indiquen leakage
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

def check_data_leakage():
    """Verificación completa de data leakage"""
    
    print("=" * 80)
    print("DIAGNÓSTICO DE DATA LEAKAGE")
    print("=" * 80)
    print()
    
    # 1. Cargar datos originales
    print("1. CARGANDO DATOS ORIGINALES")
    print("-" * 80)
    data_path = 'data/Variables_Horno.csv'
    if not os.path.exists(data_path):
        print(f"ERROR: No se encuentra {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"✓ Datos cargados: {df.shape[0]} muestras, {df.shape[1]} columnas")
    
    # Separar features y target
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    feature_names = df.columns[1:].tolist()
    
    print(f"✓ Features: {X.shape[1]}")
    print(f"✓ Target: {df.columns[0]}")
    print()
    
    # 2. Replicar el split del preprocessor
    print("2. REPLICANDO SPLIT DE DATOS (Temporal: 70% train, 30% test)")
    print("-" * 80)
    train_ratio = 0.7
    split_index = int(len(X) * train_ratio)
    
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    print(f"✓ Train: {X_train.shape[0]} muestras")
    print(f"✓ Test: {X_test.shape[0]} muestras")
    print(f"✓ Distribución train - Clase 0: {(y_train == 0).sum()}, Clase 1: {(y_train == 1).sum()}")
    print(f"✓ Distribución test  - Clase 0: {(y_test == 0).sum()}, Clase 1: {(y_test == 1).sum()}")
    print()
    
    # 3. Verificar duplicados EXACTOS entre train y test
    print("3. VERIFICANDO DUPLICADOS EXACTOS ENTRE TRAIN Y TEST")
    print("-" * 80)
    
    # Convertir a DataFrame para usar drop_duplicates
    train_df = pd.DataFrame(X_train)
    test_df = pd.DataFrame(X_test)
    
    # Buscar muestras idénticas
    duplicates_found = []
    for i, test_sample in enumerate(X_test):
        # Buscar si esta muestra existe en train
        matches = np.all(X_train == test_sample, axis=1)
        if matches.any():
            train_indices = np.where(matches)[0]
            duplicates_found.append({
                'test_idx': i + split_index,
                'train_indices': train_indices.tolist(),
                'sample': test_sample
            })
    
    if duplicates_found:
        print(f"🚨 ENCONTRADAS {len(duplicates_found)} MUESTRAS DE TEST QUE EXISTEN EN TRAIN!")
        print(f"   Esto es GRAVE y explicaría el 100% de precisión.")
        for dup in duplicates_found[:5]:  # Mostrar primeras 5
            print(f"   - Test muestra {dup['test_idx']} existe en train indices: {dup['train_indices']}")
        if len(duplicates_found) > 5:
            print(f"   ... y {len(duplicates_found) - 5} más")
    else:
        print("✓ No se encontraron muestras exactamente idénticas entre train y test")
    print()
    
    # 4. Verificar similitud muy alta (posible data leakage)
    print("4. VERIFICANDO SIMILITUD MUY ALTA ENTRE TRAIN Y TEST")
    print("-" * 80)
    
    # Calcular similitud coseno entre muestras de test y train
    high_similarity_count = 0
    similarity_threshold = 0.9999
    
    for i, test_sample in enumerate(X_test[:100]):  # Muestrear primeras 100 para velocidad
        similarities = cosine_similarity([test_sample], X_train).flatten()
        max_sim = similarities.max()
        if max_sim >= similarity_threshold:
            high_similarity_count += 1
    
    if high_similarity_count > 0:
        print(f"⚠️  {high_similarity_count} de 100 muestras de test tienen similitud > {similarity_threshold} con train")
        print(f"   Esto podría indicar data leakage")
    else:
        print(f"✓ Las muestras de test no son extremadamente similares a las de train")
    print()
    
    # 5. Verificar cómo se ajusta el scaler
    print("5. VERIFICANDO AJUSTE DEL SCALER")
    print("-" * 80)
    
    # Verificar si existe un scaler guardado
    scaler_path = 'models/scaler.pkl'
    if os.path.exists(scaler_path):
        scaler_loaded = joblib.load(scaler_path)
        print("✓ Scaler encontrado en models/scaler.pkl")
        
        # Verificar los parámetros del scaler
        mean_train = X_train.mean(axis=0)
        std_train = X_train.std(axis=0)
        
        # Comparar con los parámetros del scaler guardado
        scaler_mean = scaler_loaded.mean_
        scaler_std = scaler_loaded.scale_
        
        # Verificar si el scaler fue ajustado con TODOS los datos (leakage)
        mean_all = X.mean(axis=0)
        std_all = X.std(axis=0)
        
        diff_from_train = np.abs(scaler_mean - mean_train).max()
        diff_from_all = np.abs(scaler_mean - mean_all).max()
        
        print(f"  Diferencia máxima entre scaler.mean_ y train.mean: {diff_from_train:.10f}")
        print(f"  Diferencia máxima entre scaler.mean_ y full_data.mean: {diff_from_all:.10f}")
        
        if diff_from_all < diff_from_train * 0.1:
            print("🚨 PROBLEMA: El scaler parece haber sido ajustado con TODOS los datos")
            print("   Esto es DATA LEAKAGE - el test set influyó en el scaler")
        else:
            print("✓ El scaler parece estar ajustado solo con datos de entrenamiento")
    else:
        print("⚠️  No se encontró scaler guardado. Esto es normal si no se ha entrenado aún.")
    print()
    
    # 6. Verificar correlaciones perfectas con el target
    print("6. VERIFICANDO CORRELACIONES PERFECTAS CON EL TARGET")
    print("-" * 80)
    
    train_df_features = pd.DataFrame(X_train, columns=feature_names)
    train_y_series = pd.Series(y_train, name='POWER_ON')
    
    correlations = train_df_features.corrwith(train_y_series).abs().sort_values(ascending=False)
    
    perfect_correlations = correlations[correlations >= 0.999]
    if len(perfect_correlations) > 0:
        print(f"🚨 ENCONTRADAS {len(perfect_correlations)} FEATURES CON CORRELACIÓN ≥ 0.999")
        print("   Esto explica por qué el modelo tiene 100% de precisión:")
        for feat, corr in perfect_correlations.items():
            print(f"   - {feat}: {corr:.6f}")
    else:
        print("✓ No hay correlaciones perfectas (≥0.999)")
        print(f"   Correlación máxima: {correlations.max():.6f} ({correlations.idxmax()})")
    print()
    
    # 7. Verificar si hay features que son transformaciones del target
    print("7. VERIFICANDO SI HAY FEATURES QUE SON TRANSFORMACIONES DEL TARGET")
    print("-" * 80)
    
    target_transforms = []
    for i, col in enumerate(feature_names):
        feature_values = X_train[:, i]
        
        # Verificar si es una transformación simple del target
        # Inverso
        if np.allclose(feature_values, 1 - y_train, atol=1e-6):
            target_transforms.append((col, "inverso (1 - target)"))
        
        # Multiplicado por constante
        if len(np.unique(y_train)) == 2:
            class_0_mean = feature_values[y_train == 0].mean()
            class_1_mean = feature_values[y_train == 1].mean()
            if abs(class_0_mean - class_1_mean) > 0.5:  # Gran diferencia
                # Verificar si es aproximadamente proporcional
                if class_0_mean < 0.1 and class_1_mean > 0.7:
                    # Podría ser aproximadamente proporcional al target
                    target_transforms.append((col, f"aproximadamente proporcional (clase0={class_0_mean:.3f}, clase1={class_1_mean:.3f})"))
    
    if target_transforms:
        print(f"🚨 ENCONTRADAS {len(target_transforms)} FEATURES QUE SON TRANSFORMACIONES DEL TARGET:")
        for feat, transform_type in target_transforms:
            print(f"   - {feat}: {transform_type}")
    else:
        print("✓ No se encontraron features que sean transformaciones directas del target")
    print()
    
    # 8. Resumen y recomendaciones
    print("=" * 80)
    print("RESUMEN Y DIAGNÓSTICO")
    print("=" * 80)
    print()
    
    issues = []
    
    if duplicates_found:
        issues.append(f"🚨 CRÍTICO: {len(duplicates_found)} muestras de test existen en train")
    
    if high_similarity_count > 10:
        issues.append(f"⚠️  ALTO: {high_similarity_count}% de muestras de test muy similares a train")
    
    if os.path.exists(scaler_path):
        if diff_from_all < diff_from_train * 0.1:
            issues.append("🚨 CRÍTICO: Scaler ajustado con todos los datos (data leakage)")
    
    if len(perfect_correlations) > 0:
        issues.append(f"🚨 CRÍTICO: {len(perfect_correlations)} features con correlación perfecta con target")
    
    if target_transforms:
        issues.append(f"⚠️  ALTO: {len(target_transforms)} features son transformaciones del target")
    
    if issues:
        print("PROBLEMAS ENCONTRADOS:")
        for issue in issues:
            print(f"  {issue}")
        print()
        print("RECOMENDACIONES:")
        print("  1. Si hay duplicados: Eliminar duplicados o usar split estratificado aleatorio")
        print("  2. Si el scaler tiene leakage: Asegurarse de ajustar solo con train")
        print("  3. Si hay correlaciones perfectas: Revisar el origen de los datos")
        print("  4. Implementar cross-validation para evaluación más robusta")
    else:
        print("✓ No se encontraron problemas obvios de data leakage")
        print()
        print("Sin embargo, el 100% de precisión sigue siendo sospechoso.")
        print("Posibles causas:")
        print("  1. El dataset es sintético y las clases son perfectamente separables")
        print("  2. El problema es trivial con estas features")
        print("  3. Hay algún patrón temporal muy fuerte en los datos")
        print()
        print("RECOMENDACIONES:")
        print("  1. Implementar cross-validation estratificada")
        print("  2. Evaluar con un dataset completamente nuevo si es posible")
        print("  3. Analizar si el problema es realmente separable linealmente")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    check_data_leakage()

