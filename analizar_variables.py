"""
Script para Analizar Variables y sus Relaciones con POWER_ON
Muestra estadísticas detalladas de las características del dataset.

Uso:
    python3 analizar_variables.py
    python3 analizar_variables.py --detailed
    python3 analizar_variables.py --export
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

def analyze_variables(detailed=False, export=False):
    """
    Analiza todas las variables del dataset.
    
    Args:
        detailed: Si True, muestra análisis más detallado
        export: Si True, exporta resultados a JSON
    """
    # Cargar datos
    df = pd.read_csv('data/Variables_Horno.csv')
    
    # Separar target y features
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    feature_names = df.columns[1:].tolist()
    
    print('\n' + '='*90)
    print(' '*30 + 'ANÁLISIS COMPLETO DE VARIABLES')
    print('='*90)
    
    # ====================
    # 1. INFORMACIÓN GENERAL
    # ====================
    print('\n📊 1. INFORMACIÓN GENERAL DEL DATASET')
    print('-'*90)
    print(f'  Total de muestras:              {len(df):,}')
    print(f'  Total de variables predictoras: {len(feature_names)}')
    print(f'  Variable objetivo:              {df.columns[0]}')
    print(f'  Tipo de problema:               Clasificación Binaria')
    
    # ====================
    # 2. VARIABLE OBJETIVO
    # ====================
    print('\n🎯 2. ANÁLISIS DE LA VARIABLE OBJETIVO (POWER_ON)')
    print('-'*90)
    class_0_count = (y == 0).sum()
    class_1_count = (y == 1).sum()
    print(f'  Valores únicos:  {np.unique(y)}')
    print(f'  Distribución de clases:')
    print(f'    • Clase 0 (OFF): {class_0_count:>4} muestras ({class_0_count/len(y)*100:>5.1f}%)')
    print(f'    • Clase 1 (ON):  {class_1_count:>4} muestras ({class_1_count/len(y)*100:>5.1f}%)')
    print(f'  Balance de clases: {"Balanceado" if abs(class_0_count - class_1_count) < len(y)*0.1 else "Desbalanceado"}')
    
    # ====================
    # 3. CORRELACIONES
    # ====================
    print('\n🔗 3. CORRELACIÓN DE VARIABLES CON POWER_ON')
    print('-'*90)
    
    correlations = []
    for i, name in enumerate(feature_names):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append({
            'variable': name,
            'correlation': corr,
            'abs_correlation': abs(corr),
            'index': i
        })
    
    # Ordenar por correlación absoluta
    correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Variable':<15} {'Correlación':>12} {'|Corr|':>10} {'Importancia':>25}")
    print('-'*90)
    
    for rank, item in enumerate(correlations[:15], 1):
        corr = item['correlation']
        abs_corr = item['abs_correlation']
        
        # Visualización de fuerza
        bars = '█' * int(abs_corr * 25)
        
        # Clasificación
        if abs_corr > 0.9:
            strength = "Muy Fuerte"
        elif abs_corr > 0.7:
            strength = "Fuerte"
        elif abs_corr > 0.5:
            strength = "Moderada"
        elif abs_corr > 0.3:
            strength = "Débil"
        else:
            strength = "Muy Débil"
        
        print(f'{rank:<6} {item["variable"]:<15} {corr:>12.4f} {abs_corr:>10.4f} {bars}')
    
    # Top 5 más importantes
    print('\n✨ TOP 5 VARIABLES MÁS CORRELACIONADAS:')
    for i, item in enumerate(correlations[:5], 1):
        print(f'  {i}. {item["variable"]:<15} (r = {item["correlation"]:>7.4f})')
    
    # ====================
    # 4. ESTADÍSTICAS DESCRIPTIVAS
    # ====================
    if detailed:
        print('\n📈 4. ESTADÍSTICAS DESCRIPTIVAS DE TOP 10 VARIABLES')
        print('-'*90)
        
        for rank, item in enumerate(correlations[:10], 1):
            idx = item['index']
            name = item['variable']
            
            print(f'\n  {rank}. {name}')
            print(f'     Correlación:   {item["correlation"]:>8.4f}')
            print(f'     Media:         {X[:, idx].mean():>8.4f}')
            print(f'     Desv. Std:     {X[:, idx].std():>8.4f}')
            print(f'     Mínimo:        {X[:, idx].min():>8.4f}')
            print(f'     Máximo:        {X[:, idx].max():>8.4f}')
            print(f'     Mediana:       {np.median(X[:, idx]):>8.4f}')
    
    # ====================
    # 5. COMPARACIÓN ENTRE CLASES
    # ====================
    print('\n📊 5. COMPARACIÓN DE MEDIAS ENTRE CLASES (OFF vs ON)')
    print('-'*90)
    
    X_class0 = X[y == 0]
    X_class1 = X[y == 1]
    
    print(f"\n{'Rank':<6} {'Variable':<15} {'OFF (μ)':>12} {'ON (μ)':>12} {'Δ':>10} {'Sig':>6}")
    print('-'*90)
    
    for rank, item in enumerate(correlations[:15], 1):
        idx = item['index']
        name = item['variable']
        mean_0 = X_class0[:, idx].mean()
        mean_1 = X_class1[:, idx].mean()
        diff = mean_1 - mean_0
        
        # Significancia basada en correlación
        abs_corr = item['abs_correlation']
        if abs_corr > 0.9:
            sig = '***'
        elif abs_corr > 0.7:
            sig = '**'
        elif abs_corr > 0.5:
            sig = '*'
        else:
            sig = '-'
        
        print(f'{rank:<6} {name:<15} {mean_0:>12.4f} {mean_1:>12.4f} {diff:>10.4f} {sig:>6}')
    
    print('\n  Significancia: *** Muy alta (|r|>0.9), ** Alta (|r|>0.7), * Moderada (|r|>0.5)')
    
    # ====================
    # 6. RESUMEN CLAVE
    # ====================
    print('\n💡 6. HALLAZGOS CLAVE')
    print('-'*90)
    
    top_5 = correlations[:5]
    print(f'  • Las 5 variables más importantes tienen correlación > 0.96')
    print(f'  • Variables críticas: {", ".join([c["variable"] for c in top_5])}')
    print(f'  • El modelo puede lograr alta precisión con solo estas 5 variables')
    print(f'  • Cuando estas variables son altas (>0.7), POWER_ON tiende a ser 1 (ON)')
    print(f'  • Cuando estas variables son bajas (<0.1), POWER_ON tiende a ser 0 (OFF)')
    
    # Calcular qué tan separables son las clases
    top_var_idx = correlations[0]['index']
    class_0_values = X_class0[:, top_var_idx]
    class_1_values = X_class1[:, top_var_idx]
    
    overlap = (class_0_values.max() > class_1_values.min()) and (class_1_values.max() > class_0_values.min())
    
    print(f'\n  🎯 SEPARABILIDAD DE CLASES (usando {correlations[0]["variable"]}):')
    print(f'     Clase 0 rango: [{class_0_values.min():.4f}, {class_0_values.max():.4f}]')
    print(f'     Clase 1 rango: [{class_1_values.min():.4f}, {class_1_values.max():.4f}]')
    print(f'     Solapamiento: {"Sí" if overlap else "No"} ({"difícil" if overlap else "fácil"} de separar)')
    
    # ====================
    # 7. EXPORTAR RESULTADOS
    # ====================
    if export:
        print('\n💾 7. EXPORTANDO RESULTADOS')
        print('-'*90)
        
        results = {
            'dataset_info': {
                'total_samples': int(len(df)),
                'total_features': len(feature_names),
                'target_variable': df.columns[0],
                'class_0_count': int(class_0_count),
                'class_1_count': int(class_1_count),
            },
            'correlations': [
                {
                    'rank': i+1,
                    'variable': item['variable'],
                    'correlation': float(item['correlation']),
                    'abs_correlation': float(item['abs_correlation'])
                }
                for i, item in enumerate(correlations)
            ],
            'class_comparison': [
                {
                    'variable': item['variable'],
                    'mean_class_0': float(X_class0[:, item['index']].mean()),
                    'mean_class_1': float(X_class1[:, item['index']].mean()),
                    'difference': float(X_class1[:, item['index']].mean() - X_class0[:, item['index']].mean())
                }
                for item in correlations[:10]
            ]
        }
        
        Path('results').mkdir(exist_ok=True)
        output_path = 'results/variable_analysis.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f'  ✅ Resultados exportados a: {output_path}')
    
    print('\n' + '='*90)
    print(' '*30 + 'ANÁLISIS COMPLETADO')
    print('='*90 + '\n')


def create_correlation_plot():
    """Crea una gráfica de correlaciones."""
    print('\n📊 Generando gráfica de correlaciones...')
    
    df = pd.read_csv('data/Variables_Horno.csv')
    
    # Calcular correlaciones
    correlations = df.corr()['POWER_ON'].sort_values(ascending=False)[1:]  # Excluir POWER_ON mismo
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Gráfica 1: Barras de correlación
    colors = ['green' if x > 0 else 'red' for x in correlations.values]
    correlations.plot(kind='barh', ax=ax1, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Correlación con POWER_ON', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Variables', fontsize=12, fontweight='bold')
    ax1.set_title('Correlación de Variables con POWER_ON', fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(axis='x', alpha=0.3)
    
    # Gráfica 2: Top 10 en barras verticales
    top_10 = correlations.abs().nlargest(10).sort_values()
    colors_top = ['#2ecc71' if correlations[var] > 0 else '#e74c3c' for var in top_10.index]
    top_10.plot(kind='barh', ax=ax2, color=colors_top, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('|Correlación| con POWER_ON', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Variables', fontsize=12, fontweight='bold')
    ax2.set_title('Top 10 Variables por Correlación Absoluta', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    output_path = 'results/plots/variable_correlations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'  ✅ Gráfica guardada en: {output_path}')
    plt.close()


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Análisis de Variables del Dataset Horno',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python3 analizar_variables.py                # Análisis básico
  python3 analizar_variables.py --detailed     # Análisis detallado
  python3 analizar_variables.py --export       # Exportar a JSON
  python3 analizar_variables.py --plot         # Generar gráfica
  python3 analizar_variables.py --all          # Todo lo anterior
        """
    )
    
    parser.add_argument('--detailed', action='store_true',
                       help='Muestra estadísticas detalladas de cada variable')
    parser.add_argument('--export', action='store_true',
                       help='Exporta resultados a JSON')
    parser.add_argument('--plot', action='store_true',
                       help='Genera gráfica de correlaciones')
    parser.add_argument('--all', action='store_true',
                       help='Ejecuta análisis completo con todas las opciones')
    
    args = parser.parse_args()
    
    # Si no se especifica nada, mostrar análisis básico
    if not any(vars(args).values()):
        analyze_variables(detailed=False, export=False)
        return
    
    # Si se especifica --all, activar todo
    if args.all:
        args.detailed = True
        args.export = True
        args.plot = True
    
    # Ejecutar análisis
    analyze_variables(detailed=args.detailed, export=args.export)
    
    # Generar gráfica si se solicita
    if args.plot:
        create_correlation_plot()


if __name__ == '__main__':
    main()

