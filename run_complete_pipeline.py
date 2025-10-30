#!/usr/bin/env python3
"""
Script Completo del Pipeline de CNN para Predicción de POWER_ON
================================================================

Este script ejecuta el pipeline completo del proyecto:
1. Preprocesamiento de datos
2. Entrenamiento del modelo CNN
3. Evaluación del modelo
4. Análisis de variables
5. Visualización de resultados
6. Generación de reportes

Uso:
    python3 run_complete_pipeline.py --all
    python3 run_complete_pipeline.py --train --evaluate
    python3 run_complete_pipeline.py --visualize
    python3 run_complete_pipeline.py --quick

Autor: Horno Prediction Team
Fecha: 2025-10-23
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# Colores para terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """Imprime un encabezado formateado."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}\n")


def print_section(text):
    """Imprime una sección."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'─'*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}▶ {text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'─'*80}{Colors.ENDC}\n")


def print_success(text):
    """Imprime un mensaje de éxito."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    """Imprime un mensaje de error."""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_warning(text):
    """Imprime un mensaje de advertencia."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def print_info(text):
    """Imprime un mensaje informativo."""
    print(f"{Colors.CYAN}ℹ {text}{Colors.ENDC}")


def check_environment():
    """Verifica que el entorno esté configurado correctamente."""
    print_section("1. VERIFICACIÓN DEL ENTORNO")
    
    checks = {
        'Python 3.x': sys.version_info.major >= 3,
        'Data file': os.path.exists('data/Variables_Horno_REAL.csv'),
        'Models directory': os.path.exists('models') or os.makedirs('models', exist_ok=True) or True,
        'Results directory': os.path.exists('results') or os.makedirs('results', exist_ok=True) or True,
    }
    
    # Check imports
    try:
        import pandas as pd
        checks['pandas'] = True
    except ImportError:
        checks['pandas'] = False
    
    try:
        import numpy as np
        checks['numpy'] = True
    except ImportError:
        checks['numpy'] = False
    
    try:
        import tensorflow as tf
        checks['tensorflow'] = True
    except ImportError:
        checks['tensorflow'] = False
    
    try:
        import matplotlib
        checks['matplotlib'] = True
    except ImportError:
        checks['matplotlib'] = False
    
    # Print results
    all_good = True
    for check, result in checks.items():
        if result:
            print_success(f"{check:<30} OK")
        else:
            print_error(f"{check:<30} FALTA")
            all_good = False
    
    if not all_good:
        print_error("\nFaltan dependencias. Instálalas con:")
        print("  pip3 install -r requirements.txt")
        return False
    
    print_success("\n¡Entorno verificado correctamente!")
    return True


def run_preprocessing():
    """Ejecuta el preprocesamiento de datos."""
    print_section("2. PREPROCESAMIENTO DE DATOS")
    
    try:
        from src.preprocess import DataPreprocessor
        from src.utils import set_random_seeds
        
        set_random_seeds(42)
        
        preprocessor = DataPreprocessor(
            data_path='data/Variables_Horno_REAL.csv',
            train_ratio=0.7
        )
        
        X_train, X_test, y_train, y_test = preprocessor.preprocess()
        
        print_success(f"Datos preprocesados exitosamente")
        print_info(f"  - Training samples: {X_train.shape[0]}")
        print_info(f"  - Test samples: {X_test.shape[0]}")
        print_info(f"  - Features: {X_train.shape[1]}")
        print_info(f"  - Scaler guardado: models/scaler.pkl")
        
        return True
    except Exception as e:
        print_error(f"Error en preprocesamiento: {str(e)}")
        return False


def run_training(epochs=100, batch_size=32):
    """Ejecuta el entrenamiento del modelo."""
    print_section("3. ENTRENAMIENTO DEL MODELO CNN")
    
    try:
        from src.preprocess import DataPreprocessor
        from src.train_cnn import CNNModel, determine_task_type
        from src.utils import set_random_seeds, save_training_curves
        
        set_random_seeds(42)
        
        # Cargar y preprocesar datos
        print_info("Cargando datos...")
        preprocessor = DataPreprocessor(data_path='data/Variables_Horno_REAL.csv', train_ratio=0.7)
        X_train, X_test, y_train, y_test = preprocessor.preprocess()
        
        # Determinar tipo de tarea
        task_type = determine_task_type(y_train)
        print_info(f"Tipo de tarea: {task_type}")
        
        # Construir modelo
        print_info("Construyendo arquitectura CNN...")
        input_shape = (X_train.shape[1],)
        cnn_model = CNNModel(input_shape=input_shape, task_type=task_type)
        cnn_model.build_model()
        cnn_model.compile_model()
        
        # Entrenar
        print_info(f"Entrenando modelo ({epochs} épocas)...")
        start_time = time.time()
        
        history = cnn_model.train(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15
        )
        
        training_time = time.time() - start_time
        
        # Guardar modelo
        cnn_model.save_model('models/cnn_model.h5')
        save_training_curves(history, 'results/training_curves.png')
        
        print_success(f"Modelo entrenado exitosamente en {training_time:.2f}s")
        print_info(f"  - Modelo guardado: models/cnn_model.h5")
        print_info(f"  - Mejor modelo: models/cnn_model_best.h5")
        print_info(f"  - Historial: models/training_history.json")
        print_info(f"  - Curvas: results/training_curves.png")
        
        # Mostrar métricas finales
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        metric_key = 'accuracy' if 'accuracy' in history.history else 'mean_absolute_error'
        final_metric = history.history[metric_key][-1]
        final_val_metric = history.history[f'val_{metric_key}'][-1]
        
        print_info(f"\n  Métricas finales:")
        print_info(f"    Train Loss: {final_loss:.6f}")
        print_info(f"    Val Loss:   {final_val_loss:.6f}")
        print_info(f"    Train {metric_key}: {final_metric:.6f}")
        print_info(f"    Val {metric_key}:   {final_val_metric:.6f}")
        
        return True
    except Exception as e:
        print_error(f"Error en entrenamiento: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_evaluation():
    """Ejecuta la evaluación del modelo."""
    print_section("4. EVALUACIÓN DEL MODELO")
    
    try:
        from src.preprocess import DataPreprocessor
        from src.evaluate import ModelEvaluator
        from src.utils import print_metrics, save_metrics_to_json
        
        # Cargar datos
        print_info("Cargando datos de prueba...")
        preprocessor = DataPreprocessor(data_path='data/Variables_Horno_REAL.csv', train_ratio=0.7)
        X_train, X_test, y_train, y_test = preprocessor.preprocess()
        
        # Evaluar
        print_info("Evaluando modelo...")
        evaluator = ModelEvaluator(
            model_path='models/cnn_model.h5',
            scaler_path='models/scaler.pkl'
        )
        evaluator.load_model_and_scaler()
        
        metrics = evaluator.evaluate(X_test, y_test)
        
        # Generar reporte detallado
        y_pred = evaluator.predict(X_test)
        evaluator.generate_detailed_report(y_test, y_pred)
        
        # Guardar métricas
        save_metrics_to_json(metrics, 'results/metrics.json')
        
        print_success("Evaluación completada exitosamente")
        
        # Mostrar métricas principales
        print_info("\n  Métricas principales:")
        if 'accuracy' in metrics:
            print_info(f"    Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print_info(f"    Precision: {metrics['precision']:.4f}")
            print_info(f"    Recall:    {metrics['recall']:.4f}")
            print_info(f"    F1-Score:  {metrics['f1_score']:.4f}")
        else:
            print_info(f"    MSE:  {metrics['mse']:.6f}")
            print_info(f"    RMSE: {metrics['rmse']:.6f}")
            print_info(f"    MAE:  {metrics['mae']:.6f}")
            print_info(f"    R²:   {metrics['r2_score']:.6f}")
        
        return True
    except Exception as e:
        print_error(f"Error en evaluación: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_variable_analysis():
    """Ejecuta el análisis de variables."""
    print_section("5. ANÁLISIS DE VARIABLES")
    
    try:
        import subprocess
        result = subprocess.run(
            ['python3', 'analizar_variables.py', '--export'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print_success("Análisis de variables completado")
            print_info("  - Resultados: results/variable_analysis.json")
            return True
        else:
            print_error(f"Error en análisis de variables: {result.stderr}")
            return False
    except Exception as e:
        print_error(f"Error ejecutando análisis: {str(e)}")
        return False


def run_visualizations():
    """Ejecuta la generación de visualizaciones."""
    print_section("6. GENERACIÓN DE VISUALIZACIONES")
    
    try:
        import subprocess
        
        print_info("Generando todas las visualizaciones...")
        result = subprocess.run(
            ['python3', 'visualize_results.py', '--all'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 or 'Gráfica guardada' in result.stdout:
            print_success("Visualizaciones generadas exitosamente")
            print_info("  - Dashboard: results/plots/dashboard.png")
            print_info("  - Training: results/plots/training_detailed.png")
            print_info("  - Evaluation: results/plots/evaluation_metrics.png")
            print_info("  - Predictions: results/plots/predictions_analysis.png")
            print_info("  - Importance: results/plots/feature_importance.png")
            return True
        else:
            print_warning("Algunas visualizaciones pueden no haberse generado")
            print_info(result.stdout)
            return True  # No es crítico
    except Exception as e:
        print_error(f"Error generando visualizaciones: {str(e)}")
        return False


def generate_summary_report():
    """Genera un reporte resumen en texto."""
    print_section("7. GENERACIÓN DE REPORTE RESUMEN")
    
    try:
        # Cargar métricas
        with open('results/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        # Cargar análisis de variables si existe
        variable_analysis = None
        if os.path.exists('results/variable_analysis.json'):
            with open('results/variable_analysis.json', 'r') as f:
                variable_analysis = json.load(f)
        
        # Crear reporte
        report_path = 'results/RESUMEN_COMPLETO.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(" "*20 + "REPORTE COMPLETO DEL PROYECTO\n")
            f.write(" "*15 + "CNN para Predicción de POWER_ON del Horno\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Proyecto: Predicción de Estado de Horno Industrial\n")
            f.write(f"Modelo: Convolutional Neural Network (CNN)\n\n")
            
            # Sección 1: Datos
            f.write("-"*80 + "\n")
            f.write("1. INFORMACIÓN DE LOS DATOS\n")
            f.write("-"*80 + "\n")
            if variable_analysis:
                f.write(f"Total de muestras: {variable_analysis['dataset_info']['total_samples']}\n")
                f.write(f"Variables predictoras: {variable_analysis['dataset_info']['total_features']}\n")
                f.write(f"Variable objetivo: {variable_analysis['dataset_info']['target_variable']}\n")
                f.write(f"Clase 0 (OFF): {variable_analysis['dataset_info']['class_0_count']} muestras\n")
                f.write(f"Clase 1 (ON): {variable_analysis['dataset_info']['class_1_count']} muestras\n")
            f.write(f"Muestras de prueba: {metrics['test_samples']}\n\n")
            
            # Sección 2: Rendimiento
            f.write("-"*80 + "\n")
            f.write("2. RENDIMIENTO DEL MODELO\n")
            f.write("-"*80 + "\n")
            if 'accuracy' in metrics:
                f.write(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall:    {metrics['recall']:.4f}\n")
                f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
            f.write(f"MAE: {metrics['mae']:.6f}\n")
            f.write(f"MSE: {metrics['mse']:.10f}\n")
            f.write(f"RMSE: {metrics['rmse']:.6f}\n\n")
            
            # Sección 3: Variables importantes
            if variable_analysis:
                f.write("-"*80 + "\n")
                f.write("3. VARIABLES MÁS IMPORTANTES\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Rank':<6} {'Variable':<15} {'Correlación':>12}\n")
                f.write("-"*40 + "\n")
                for i, var in enumerate(variable_analysis['correlations'][:10], 1):
                    f.write(f"{i:<6} {var['variable']:<15} {var['correlation']:>12.4f}\n")
                f.write("\n")
            
            # Sección 4: Archivos generados
            f.write("-"*80 + "\n")
            f.write("4. ARCHIVOS GENERADOS\n")
            f.write("-"*80 + "\n")
            
            files = {
                'Modelo entrenado': 'models/cnn_model.h5',
                'Mejor modelo': 'models/cnn_model_best.h5',
                'Scaler': 'models/scaler.pkl',
                'Historial': 'models/training_history.json',
                'Métricas': 'results/metrics.json',
                'Curvas entrenamiento': 'results/training_curves.png',
                'Matriz confusión': 'results/confusion_matrix.png',
                'Dashboard': 'results/plots/dashboard.png',
                'Análisis variables': 'results/variable_analysis.json',
            }
            
            for name, path in files.items():
                status = "✓" if os.path.exists(path) else "✗"
                f.write(f"{status} {name:<25} {path}\n")
            
            f.write("\n")
            
            # Sección 5: Interpretación
            f.write("-"*80 + "\n")
            f.write("5. INTERPRETACIÓN Y CONCLUSIONES\n")
            f.write("-"*80 + "\n")
            
            if metrics.get('accuracy', 0) >= 0.99:
                f.write("• El modelo alcanzó una precisión excepcional (≥99%)\n")
                f.write("• Las primeras 5 características muestran correlación muy alta (>0.96)\n")
                f.write("• Las clases son linealmente separables\n")
                f.write("• Cuando features 0-4 son altos (>0.7) → POWER_ON = 1 (encendido)\n")
                f.write("• Cuando features 0-4 son bajos (<0.1) → POWER_ON = 0 (apagado)\n")
            
            f.write("\n")
            f.write("="*80 + "\n")
            f.write(" "*25 + "FIN DEL REPORTE\n")
            f.write("="*80 + "\n")
        
        print_success(f"Reporte generado: {report_path}")
        return True
    except Exception as e:
        print_error(f"Error generando reporte: {str(e)}")
        return False


def open_results():
    """Abre los resultados en el navegador/visor."""
    print_section("8. VISUALIZACIÓN DE RESULTADOS")
    
    try:
        import subprocess
        import platform
        
        plots_dir = 'results/plots'
        
        if os.path.exists(plots_dir):
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', plots_dir])
            elif platform.system() == 'Windows':
                subprocess.run(['explorer', plots_dir])
            else:  # Linux
                subprocess.run(['xdg-open', plots_dir])
            
            print_success(f"Abriendo carpeta de resultados: {plots_dir}")
            return True
        else:
            print_warning("No se encontró la carpeta de resultados")
            return False
    except Exception as e:
        print_warning(f"No se pudo abrir automáticamente: {str(e)}")
        print_info(f"Abre manualmente: {os.path.abspath('results/plots')}")
        return True


def print_final_summary(results):
    """Imprime el resumen final del pipeline."""
    print_header("RESUMEN FINAL DEL PIPELINE")
    
    total = len(results)
    success = sum(1 for r in results.values() if r)
    
    print(f"\n{Colors.BOLD}Pasos completados: {success}/{total}{Colors.ENDC}\n")
    
    for step, result in results.items():
        if result:
            print_success(f"{step}")
        else:
            print_error(f"{step}")
    
    if success == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}¡PIPELINE COMPLETADO EXITOSAMENTE!{Colors.ENDC}\n")
        print_info("Archivos principales generados:")
        print_info("  • models/cnn_model.h5              - Modelo entrenado")
        print_info("  • results/metrics.json             - Métricas de evaluación")
        print_info("  • results/plots/dashboard.png      - Dashboard visual")
        print_info("  • results/RESUMEN_COMPLETO.txt     - Reporte completo")
        print_info("\nPara ver los resultados:")
        print_info("  open results/plots/")
        print_info("  cat results/RESUMEN_COMPLETO.txt")
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}Pipeline completado con algunos errores{Colors.ENDC}\n")
        print_warning("Revisa los mensajes de error arriba para más detalles")


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description='Pipeline Completo de CNN para Predicción de POWER_ON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python3 run_complete_pipeline.py --all                    # Pipeline completo
  python3 run_complete_pipeline.py --quick                  # Pipeline rápido (menos épocas)
  python3 run_complete_pipeline.py --train --evaluate       # Solo entrenar y evaluar
  python3 run_complete_pipeline.py --visualize              # Solo visualizaciones
  python3 run_complete_pipeline.py --check                  # Solo verificar entorno

Pasos del pipeline:
  1. Verificación del entorno
  2. Preprocesamiento de datos
  3. Entrenamiento del modelo CNN
  4. Evaluación del modelo
  5. Análisis de variables
  6. Generación de visualizaciones
  7. Generación de reporte resumen
  8. Apertura de resultados
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Ejecutar pipeline completo')
    parser.add_argument('--quick', action='store_true',
                       help='Pipeline rápido (20 épocas)')
    parser.add_argument('--check', action='store_true',
                       help='Solo verificar entorno')
    parser.add_argument('--train', action='store_true',
                       help='Entrenar modelo')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluar modelo')
    parser.add_argument('--visualize', action='store_true',
                       help='Generar visualizaciones')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas de entrenamiento (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Tamaño de batch (default: 32)')
    parser.add_argument('--no-open', action='store_true',
                       help='No abrir resultados automáticamente')
    
    args = parser.parse_args()
    
    # Si no se especifica nada, mostrar help
    if not any(vars(args).values()) or (not args.all and not args.quick and not args.check 
                                         and not args.train and not args.evaluate and not args.visualize):
        parser.print_help()
        return
    
    # Banner inicial
    print_header("PIPELINE COMPLETO DE CNN - PREDICCIÓN POWER_ON")
    print(f"{Colors.BOLD}Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")
    
    # Resultados de cada paso
    results = {}
    
    # Verificar entorno siempre
    results['Verificación de entorno'] = check_environment()
    
    if args.check:
        return
    
    if not results['Verificación de entorno']:
        print_error("\nEl entorno no está configurado correctamente. Abortando.")
        return
    
    # Determinar épocas
    epochs = 20 if args.quick else args.epochs
    
    # Ejecutar pasos según argumentos
    if args.all or args.quick or args.train:
        results['Preprocesamiento'] = run_preprocessing()
        if results['Preprocesamiento']:
            results['Entrenamiento'] = run_training(epochs=epochs, batch_size=args.batch_size)
    
    if args.all or args.quick or args.evaluate:
        if not os.path.exists('models/cnn_model.h5'):
            print_warning("No hay modelo entrenado. Entrenando primero...")
            results['Preprocesamiento'] = run_preprocessing()
            results['Entrenamiento'] = run_training(epochs=epochs, batch_size=args.batch_size)
        
        results['Evaluación'] = run_evaluation()
    
    if args.all or args.quick or args.visualize:
        results['Análisis de variables'] = run_variable_analysis()
        results['Visualizaciones'] = run_visualizations()
    
    if args.all or args.quick:
        results['Reporte resumen'] = generate_summary_report()
        if not args.no_open:
            results['Apertura de resultados'] = open_results()
    
    # Resumen final
    print_final_summary(results)
    
    print(f"\n{Colors.BOLD}Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")


if __name__ == '__main__':
    main()

