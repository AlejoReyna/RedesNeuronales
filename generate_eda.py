#!/usr/bin/env python3
"""
EDA Automation Script
======================
Generates basic exploratory analysis outputs for the Variables_Horno dataset.

Outputs saved to results/eda/:
- eda_summary.json: dataset statistics
- target_distribution.png
- correlation_heatmap.png
- top_features_swarm.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocess import DataPreprocessor


def run_eda(dataset_path: str = 'data/Variables_Horno_REAL.csv') -> None:
    """Run EDA and save plots/statistics."""
    output_dir = Path('results/eda')
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = DataPreprocessor(data_path=dataset_path)
    df = preprocessor.load_data()

    target_col = df.columns[0]

    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'target': target_col,
        'target_distribution': df[target_col].value_counts().to_dict() if df[target_col].nunique() <= 32 else 'continuous',
        'missing_values': df.isnull().sum().to_dict(),
        'describe': df.describe().to_dict(),
    }

    with open(output_dir / 'eda_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    sns.set_theme(style='whitegrid')

    target_unique = df[target_col].nunique()
    is_classification = target_unique <= 10 and set(df[target_col].unique()).issubset({0, 1})

    plt.figure(figsize=(8, 4))
    if is_classification:
        sns.countplot(x=target_col, data=df)
        plt.title('Distribución de POWER_ON')
    else:
        sns.histplot(df[target_col], kde=True, bins=30)
        plt.title(f'Distribución de {target_col}')
    plt.savefig(output_dir / 'target_distribution.png', bbox_inches='tight')
    plt.close()

    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', square=True)
    plt.title('Matriz de correlación')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', bbox_inches='tight')
    plt.close()

    top_features = corr[target_col].drop(target_col).abs().sort_values(ascending=False).head(5).index
    if is_classification:
        long_df = df.melt(id_vars=target_col, value_vars=top_features, var_name='feature', value_name='value')
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=long_df, x='feature', y='value', hue=target_col)
        plt.title('Top 5 features vs POWER_ON')
        plt.xticks(rotation=45)
        plt.savefig(output_dir / 'top_features_boxplot.png', bbox_inches='tight')
        plt.close()
    else:
        plt.figure(figsize=(10, 6))
        for feature in top_features:
            sns.scatterplot(x=df[feature], y=df[target_col], label=feature)
        plt.legend()
        plt.title('Top 5 features vs Target')
        plt.savefig(output_dir / 'top_features_scatter.png', bbox_inches='tight')
        plt.close()

    print(f"EDA outputs saved to {output_dir.resolve()}")


if __name__ == '__main__':
    run_eda()


