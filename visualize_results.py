"""
Script de Visualización de Resultados del Modelo CNN
Genera gráficas detalladas del entrenamiento, evaluación y predicciones.

Uso:
    python3 visualize_results.py --all
    python3 visualize_results.py --training
    python3 visualize_results.py --evaluation
    python3 visualize_results.py --predictions
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import joblib

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def ensure_output_dir():
    """Crea el directorio de salida si no existe."""
    os.makedirs('results/plots', exist_ok=True)


def plot_training_history(history_path='models/training_history.json', save_path='results/plots/training_detailed.png'):
    """
    Crea gráficas detalladas del historial de entrenamiento.
    
    Args:
        history_path: Ruta al archivo JSON con el historial
        save_path: Ruta para guardar la gráfica
    """
    print("\n" + "=" * 60)
    print("GENERANDO GRÁFICAS DE ENTRENAMIENTO")
    print("=" * 60)
    
    if not os.path.exists(history_path):
        print(f"❌ No se encontró {history_path}")
        print("Ejecuta primero: python3 main.py --train")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Loss principal
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(history['loss']) + 1)
    ax1.plot(epochs, history['loss'], 'b-o', label='Training Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.set_title('Evolución del Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    metric_key = 'accuracy' if 'accuracy' in history else 'mean_absolute_error'
    metric_label = 'Accuracy' if 'accuracy' in history else 'MAE'
    ax2.plot(epochs, history[metric_key], 'b-o', label=f'Training {metric_label}', linewidth=2, markersize=4)
    ax2.plot(epochs, history[f'val_{metric_key}'], 'r-s', label=f'Validation {metric_label}', linewidth=2, markersize=4)
    ax2.set_xlabel('Época')
    ax2.set_ylabel(metric_label)
    ax2.set_title(f'Evolución de {metric_label}', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Loss en escala logarítmica
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogy(epochs, history['loss'], 'b-o', label='Training Loss', linewidth=2, markersize=4)
    ax3.semilogy(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    ax3.set_xlabel('Época')
    ax3.set_ylabel('Loss (escala log)')
    ax3.set_title('Loss en Escala Logarítmica', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Diferencia entre Training y Validation Loss (Overfitting)
    ax4 = fig.add_subplot(gs[1, 1])
    diff = np.array(history['val_loss']) - np.array(history['loss'])
    ax4.plot(epochs, diff, 'g-o', linewidth=2, markersize=4)
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax4.fill_between(epochs, 0, diff, where=(diff > 0), alpha=0.3, color='red', label='Overfitting')
    ax4.fill_between(epochs, 0, diff, where=(diff <= 0), alpha=0.3, color='green', label='Good fit')
    ax4.set_xlabel('Época')
    ax4.set_ylabel('Val Loss - Train Loss')
    ax4.set_title('Análisis de Overfitting', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Learning rate (si está disponible)
    ax5 = fig.add_subplot(gs[2, 0])
    if 'lr' in history:
        ax5.plot(epochs, history['lr'], 'purple', linewidth=2)
        ax5.set_xlabel('Época')
        ax5.set_ylabel('Learning Rate')
        ax5.set_title('Evolución del Learning Rate', fontsize=14, fontweight='bold')
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Learning rate no disponible', 
                ha='center', va='center', fontsize=12)
        ax5.axis('off')
    
    # 6. Estadísticas finales
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_train_metric = history[metric_key][-1]
    final_val_metric = history[f'val_{metric_key}'][-1]
    best_val_loss = min(history['val_loss'])
    best_epoch = history['val_loss'].index(best_val_loss) + 1
    
    stats_text = f"""
    ESTADÍSTICAS FINALES
    {'='*30}
    
    Mejor Época: {best_epoch}
    Mejor Val Loss: {best_val_loss:.4f}
    
    Época Final: {len(epochs)}
    Train Loss: {final_train_loss:.4f}
    Val Loss: {final_val_loss:.4f}
    
    Train {metric_label}: {final_train_metric:.4f}
    Val {metric_label}: {final_val_metric:.4f}
    
    Diferencia Loss: {final_val_loss - final_train_loss:.4f}
    {'Overfitting detectado' if final_val_loss > final_train_loss * 1.1 else 'Buen ajuste'}
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('Análisis Detallado del Entrenamiento', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfica guardada en: {save_path}")
    plt.close()


def plot_evaluation_metrics(metrics_path='results/metrics.json', save_path='results/plots/evaluation_metrics.png'):
    """
    Visualiza las métricas de evaluación del modelo.
    
    Args:
        metrics_path: Ruta al archivo JSON con métricas
        save_path: Ruta para guardar la gráfica
    """
    print("\n" + "=" * 60)
    print("GENERANDO GRÁFICAS DE EVALUACIÓN")
    print("=" * 60)
    
    if not os.path.exists(metrics_path):
        print(f"❌ No se encontró {metrics_path}")
        print("Ejecuta primero: python3 main.py --evaluate")
        return
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Métricas principales en barras
    ax1 = axes[0, 0]
    if 'accuracy' in metrics:
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score']
        ]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = ax1.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylim([0, 1])
        ax1.set_ylabel('Valor')
        ax1.set_title('Métricas de Clasificación', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        # Métricas de regresión
        metric_names = ['MSE', 'RMSE', 'MAE', 'R²']
        metric_values = [
            metrics['mse'],
            metrics['rmse'],
            metrics['mae'],
            metrics['r2_score']
        ]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = ax1.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Valor')
        ax1.set_title('Métricas de Regresión', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Matriz de confusión (si es clasificación)
    ax2 = axes[0, 1]
    if 'confusion_matrix' in metrics:
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, 
                   cbar_kws={'label': 'Cantidad'})
        ax2.set_xlabel('Predicción')
        ax2.set_ylabel('Real')
        ax2.set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
        
        # Calcular porcentajes
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j+0.5, i+0.7, f'({cm_norm[i, j]*100:.1f}%)',
                        ha='center', va='center', fontsize=9, color='gray')
    else:
        ax2.text(0.5, 0.5, 'No aplicable para regresión', 
                ha='center', va='center', fontsize=12)
        ax2.axis('off')
    
    # 3. Comparación con baseline (si existe)
    ax3 = axes[1, 0]
    if 'baseline_comparison' in metrics and metrics['baseline_comparison']:
        comparison = metrics['baseline_comparison']
        models = ['Modelo CNN', 'Baseline']
        metric_to_plot = 'accuracy' if 'accuracy' in metrics else 'mse'
        values = [metrics[metric_to_plot], comparison.get(metric_to_plot, 0)]
        
        colors = ['#2ecc71', '#e74c3c']
        bars = ax3.bar(models, values, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Valor')
        ax3.set_title(f'Comparación: {metric_to_plot.upper()}', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Mostrar mejora
        improvement = ((values[0] - values[1]) / values[1]) * 100
        ax3.text(0.5, 0.95, f'Mejora: {improvement:+.1f}%',
                transform=ax3.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                fontsize=11, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No hay comparación con baseline', 
                ha='center', va='center', fontsize=12)
        ax3.axis('off')
    
    # 4. Resumen de métricas
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "RESUMEN DE MÉTRICAS\n" + "="*30 + "\n\n"
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and key != 'confusion_matrix':
            summary_text += f"{key.upper():.<25} {value:.4f}\n"
    
    ax4.text(0.1, 0.9, summary_text, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax4.transAxes)
    
    plt.suptitle('Análisis de Evaluación del Modelo', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfica guardada en: {save_path}")
    plt.close()


def plot_predictions_analysis(data_path='data/Variables_Horno.csv', 
                              model_path='models/cnn_model.h5',
                              scaler_path='models/scaler.pkl',
                              save_path='results/plots/predictions_analysis.png'):
    """
    Analiza las predicciones del modelo vs valores reales.
    
    Args:
        data_path: Ruta a los datos
        model_path: Ruta al modelo entrenado
        scaler_path: Ruta al scaler
        save_path: Ruta para guardar la gráfica
    """
    print("\n" + "=" * 60)
    print("GENERANDO ANÁLISIS DE PREDICCIONES")
    print("=" * 60)
    
    try:
        from src.train_cnn import CNNModel
        from tensorflow import keras
        import tensorflow as tf
    except ImportError:
        print("❌ TensorFlow no disponible. Activa venv311.")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ No se encontró el modelo en {model_path}")
        print("Ejecuta primero: python3 main.py --train")
        return
    
    # Cargar datos
    df = pd.read_csv(data_path)
    
    # Separar características y objetivo
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    
    # Split temporal (igual que en el entrenamiento)
    split_idx = int(len(X) * 0.7)
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    # Cargar scaler y escalar
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)
    
    # Cargar modelo y predecir
    model = keras.models.load_model(model_path)
    y_pred = model.predict(X_test_scaled, verbose=0)
    y_pred = y_pred.flatten()
    
    # Determinar si es clasificación o regresión
    is_classification = len(np.unique(y_test)) <= 10
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    if is_classification:
        # Clasificación binaria
        y_pred_class = (y_pred > 0.5).astype(int)
        
        # 1. Distribución de probabilidades por clase
        ax1 = fig.add_subplot(gs[0, 0])
        class_0_probs = y_pred[y_test == 0]
        class_1_probs = y_pred[y_test == 1]
        ax1.hist(class_0_probs, bins=30, alpha=0.6, label='Clase 0 (Real)', color='blue', edgecolor='black')
        ax1.hist(class_1_probs, bins=30, alpha=0.6, label='Clase 1 (Real)', color='red', edgecolor='black')
        ax1.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Threshold')
        ax1.set_xlabel('Probabilidad Predicha')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución de Probabilidades', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Matriz de confusión visual
        ax2 = fig.add_subplot(gs[0, 1])
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred_class)
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax2)
        ax2.set_xlabel('Predicción')
        ax2.set_ylabel('Real')
        ax2.set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
        
        # 3. ROC Curve
        ax3 = fig.add_subplot(gs[0, 2])
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('Curva ROC', fontsize=14, fontweight='bold')
        ax3.legend(loc="lower right")
        ax3.grid(True, alpha=0.3)
        
        # 4. Predicciones a lo largo del tiempo
        ax4 = fig.add_subplot(gs[1, :])
        indices = np.arange(len(y_test))
        ax4.scatter(indices[y_test == 0], y_pred[y_test == 0], 
                   alpha=0.6, s=30, c='blue', label='Clase 0 (Real)', edgecolors='black', linewidth=0.5)
        ax4.scatter(indices[y_test == 1], y_pred[y_test == 1], 
                   alpha=0.6, s=30, c='red', label='Clase 1 (Real)', edgecolors='black', linewidth=0.5)
        ax4.axhline(y=0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Threshold')
        ax4.set_xlabel('Índice de Muestra (Temporal)')
        ax4.set_ylabel('Probabilidad Predicha')
        ax4.set_title('Predicciones a lo Largo del Tiempo', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
    else:
        # Regresión
        # 1. Scatter plot real vs predicho
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(y_test, y_pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
        ax1.set_xlabel('Valor Real')
        ax1.set_ylabel('Valor Predicho')
        ax1.set_title('Real vs Predicho', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuos
        ax2 = fig.add_subplot(gs[0, 1])
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Valor Predicho')
        ax2.set_ylabel('Residuo')
        ax2.set_title('Análisis de Residuos', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribución de residuos
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='r', linestyle='--', lw=2)
        ax3.set_xlabel('Residuo')
        ax3.set_ylabel('Frecuencia')
        ax3.set_title('Distribución de Residuos', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Serie temporal de predicciones
        ax4 = fig.add_subplot(gs[1, :])
        indices = np.arange(len(y_test))
        ax4.plot(indices, y_test, 'b-', label='Real', alpha=0.7, linewidth=2)
        ax4.plot(indices, y_pred, 'r-', label='Predicho', alpha=0.7, linewidth=2)
        ax4.fill_between(indices, y_test, y_pred, alpha=0.3, color='gray')
        ax4.set_xlabel('Índice de Muestra (Temporal)')
        ax4.set_ylabel('Valor')
        ax4.set_title('Comparación Temporal', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Análisis de Predicciones del Modelo', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfica guardada en: {save_path}")
    plt.close()


def plot_feature_importance(data_path='data/Variables_Horno.csv',
                           model_path='models/cnn_model.h5',
                           scaler_path='models/scaler.pkl',
                           save_path='results/plots/feature_importance.png'):
    """
    Analiza la importancia de las características usando permutación.
    
    Args:
        data_path: Ruta a los datos
        model_path: Ruta al modelo
        scaler_path: Ruta al scaler
        save_path: Ruta para guardar la gráfica
    """
    print("\n" + "=" * 60)
    print("GENERANDO ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS")
    print("=" * 60)
    
    try:
        from tensorflow import keras
    except ImportError:
        print("❌ TensorFlow no disponible. Activa venv311.")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ No se encontró el modelo en {model_path}")
        return
    
    # Cargar datos
    df = pd.read_csv(data_path)
    feature_names = df.columns[1:].tolist()
    
    # Separar y preparar datos de test
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    split_idx = int(len(X) * 0.7)
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    # Cargar scaler y modelo
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)
    model = keras.models.load_model(model_path)
    
    # Calcular baseline score
    baseline_score = model.evaluate(X_test_scaled, y_test, verbose=0)[0]
    
    # Permutation importance
    print("Calculando importancia de características (puede tomar tiempo)...")
    importances = []
    
    for i in range(X_test_scaled.shape[1]):
        X_permuted = X_test_scaled.copy()
        np.random.shuffle(X_permuted[:, i])
        permuted_score = model.evaluate(X_permuted, y_test, verbose=0)[0]
        importance = permuted_score - baseline_score
        importances.append(importance)
    
    # Crear gráfica
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Ordenar características por importancia
    indices = np.argsort(importances)[::-1]
    sorted_importances = np.array(importances)[indices]
    sorted_names = np.array(feature_names)[indices]
    
    # 1. Barras horizontales
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(importances)))
    ax1.barh(range(len(importances)), sorted_importances, color=colors, edgecolor='black')
    ax1.set_yticks(range(len(importances)))
    ax1.set_yticklabels(sorted_names, fontsize=8)
    ax1.set_xlabel('Cambio en Loss (mayor = más importante)')
    ax1.set_title('Importancia de Características por Permutación', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # 2. Top 10 características
    top_n = min(10, len(importances))
    ax2.bar(range(top_n), sorted_importances[:top_n], 
           color=colors[:top_n], edgecolor='black', alpha=0.7)
    ax2.set_xticks(range(top_n))
    ax2.set_xticklabels(sorted_names[:top_n], rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Cambio en Loss')
    ax2.set_title(f'Top {top_n} Características Más Importantes', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Añadir valores en las barras
    for i, v in enumerate(sorted_importances[:top_n]):
        ax2.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfica guardada en: {save_path}")
    plt.close()


def create_summary_dashboard(save_path='results/plots/dashboard.png'):
    """
    Crea un dashboard resumido con las gráficas más importantes.
    
    Args:
        save_path: Ruta para guardar el dashboard
    """
    print("\n" + "=" * 60)
    print("GENERANDO DASHBOARD RESUMEN")
    print("=" * 60)
    
    # Verificar qué archivos existen
    files = {
        'history': 'models/training_history.json',
        'metrics': 'results/metrics.json',
        'model': 'models/cnn_model.h5'
    }
    
    available = {k: os.path.exists(v) for k, v in files.items()}
    
    if not any(available.values()):
        print("❌ No hay datos disponibles para el dashboard")
        print("Ejecuta primero: python3 main.py --train --evaluate")
        return
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # Título principal
    fig.suptitle('Dashboard de Resultados del Modelo CNN - POWER ON Prediction', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Training Loss (si disponible)
    if available['history']:
        with open(files['history'], 'r') as f:
            history = json.load(f)
        
        ax1 = fig.add_subplot(gs[0, 0])
        epochs = range(1, len(history['loss']) + 1)
        ax1.plot(epochs, history['loss'], 'b-', label='Train', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.set_title('Evolución del Loss', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy/Metric
        ax2 = fig.add_subplot(gs[0, 1])
        metric_key = 'accuracy' if 'accuracy' in history else 'mean_absolute_error'
        ax2.plot(epochs, history[metric_key], 'b-', label='Train', linewidth=2)
        ax2.plot(epochs, history[f'val_{metric_key}'], 'r-', label='Val', linewidth=2)
        ax2.set_xlabel('Época')
        ax2.set_ylabel(metric_key.replace('_', ' ').title())
        ax2.set_title('Evolución de Métrica', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Overfitting analysis
        ax3 = fig.add_subplot(gs[0, 2])
        diff = np.array(history['val_loss']) - np.array(history['loss'])
        ax3.plot(epochs, diff, 'g-', linewidth=2)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.fill_between(epochs, 0, diff, where=(diff > 0), alpha=0.3, color='red')
        ax3.fill_between(epochs, 0, diff, where=(diff <= 0), alpha=0.3, color='green')
        ax3.set_xlabel('Época')
        ax3.set_ylabel('Val Loss - Train Loss')
        ax3.set_title('Análisis de Overfitting', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # 4-6. Métricas de evaluación (si disponible)
    if available['metrics']:
        with open(files['metrics'], 'r') as f:
            metrics = json.load(f)
        
        # 4. Métricas principales
        ax4 = fig.add_subplot(gs[1, 0])
        if 'accuracy' in metrics:
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
            metric_values = [metrics['accuracy'], metrics['precision'], 
                           metrics['recall'], metrics['f1_score']]
        else:
            metric_names = ['MSE', 'RMSE', 'MAE', 'R²']
            metric_values = [metrics['mse'], metrics['rmse'], 
                           metrics['mae'], metrics['r2_score']]
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = ax4.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Valor')
        ax4.set_title('Métricas de Evaluación', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 5. Matriz de confusión (si es clasificación)
        ax5 = fig.add_subplot(gs[1, 1])
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5, cbar=False)
            ax5.set_xlabel('Predicción')
            ax5.set_ylabel('Real')
            ax5.set_title('Matriz de Confusión', fontsize=12, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'Regresión\n(No matriz de confusión)', 
                    ha='center', va='center', fontsize=11)
            ax5.axis('off')
        
        # 6. Resumen de métricas
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        summary_text = "RESUMEN\n" + "="*20 + "\n\n"
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 
                      'mse', 'rmse', 'mae', 'r2_score']
        for key in key_metrics:
            if key in metrics:
                summary_text += f"{key.upper()}: {metrics[key]:.4f}\n"
        ax6.text(0.1, 0.9, summary_text, fontsize=10, family='monospace',
                verticalalignment='top', transform=ax6.transAxes)
    
    # 7-9. Información adicional
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    info_text = "INFORMACIÓN DEL PROYECTO\n" + "="*80 + "\n\n"
    info_text += f"Proyecto: Predicción POWER ON del Horno\n"
    info_text += f"Modelo: CNN (Convolutional Neural Network)\n"
    info_text += f"Datos: Variables_Horno.csv\n"
    info_text += f"Split: 70% Train / 30% Test (temporal)\n\n"
    
    if available['history']:
        info_text += f"Épocas entrenadas: {len(history['loss'])}\n"
        info_text += f"Mejor época: {history['val_loss'].index(min(history['val_loss'])) + 1}\n"
        info_text += f"Mejor val loss: {min(history['val_loss']):.4f}\n\n"
    
    info_text += "Archivos generados:\n"
    for name, path in [('Modelo', 'models/cnn_model.h5'),
                       ('Scaler', 'models/scaler.pkl'),
                       ('Historial', 'models/training_history.json'),
                       ('Métricas', 'results/metrics.json')]:
        status = "✅" if os.path.exists(path) else "❌"
        info_text += f"  {status} {name}: {path}\n"
    
    ax7.text(0.05, 0.9, info_text, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax7.transAxes)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Dashboard guardado en: {save_path}")
    plt.close()


def main():
    """Función principal del script de visualización."""
    parser = argparse.ArgumentParser(
        description='Visualización de Resultados del Modelo CNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python3 visualize_results.py --all              # Generar todas las gráficas
  python3 visualize_results.py --training         # Solo gráficas de entrenamiento
  python3 visualize_results.py --evaluation       # Solo gráficas de evaluación
  python3 visualize_results.py --predictions      # Solo análisis de predicciones
  python3 visualize_results.py --dashboard        # Solo dashboard resumen
  python3 visualize_results.py --importance       # Solo importancia de características
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Generar todas las visualizaciones')
    parser.add_argument('--training', action='store_true',
                       help='Gráficas de entrenamiento')
    parser.add_argument('--evaluation', action='store_true',
                       help='Gráficas de evaluación')
    parser.add_argument('--predictions', action='store_true',
                       help='Análisis de predicciones')
    parser.add_argument('--importance', action='store_true',
                       help='Importancia de características')
    parser.add_argument('--dashboard', action='store_true',
                       help='Dashboard resumen')
    
    args = parser.parse_args()
    
    # Si no se especifica nada, mostrar help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Crear directorio de salida
    ensure_output_dir()
    
    print("\n" + "="*60)
    print(" "*15 + "VISUALIZACIÓN DE RESULTADOS")
    print("="*60)
    
    # Ejecutar visualizaciones solicitadas
    if args.all or args.training:
        plot_training_history()
    
    if args.all or args.evaluation:
        plot_evaluation_metrics()
    
    if args.all or args.predictions:
        plot_predictions_analysis()
    
    if args.all or args.importance:
        plot_feature_importance()
    
    if args.all or args.dashboard:
        create_summary_dashboard()
    
    print("\n" + "="*60)
    print("VISUALIZACIÓN COMPLETADA")
    print("="*60)
    print("\nGráficas generadas en: results/plots/")
    print("\nPara ver las gráficas:")
    print("  open results/plots/")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()


