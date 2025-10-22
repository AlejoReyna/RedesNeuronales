#!/usr/bin/env python3
"""
Data Leakage Analysis Script for HornoPrediction Project
This script performs comprehensive checks for data leakage and suspicious patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def analyze_data_leakage():
    """Comprehensive data leakage analysis"""
    print("=" * 80)
    print("DATA LEAKAGE ANALYSIS")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv('data/Variables_Horno.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Separate features and target
    target = df['POWER_ON']
    features = df.drop('POWER_ON', axis=1)
    
    print(f"\nTarget variable statistics:")
    print(f"Mean: {target.mean():.4f}")
    print(f"Std: {target.std():.4f}")
    print(f"Min: {target.min():.4f}")
    print(f"Max: {target.max():.4f}")
    print(f"Unique values: {sorted(target.unique())}")
    
    # 1. Check for perfect correlations
    print("\n" + "="*60)
    print("1. CHECKING FOR PERFECT CORRELATIONS")
    print("="*60)
    
    # Calculate correlations
    corr_matrix = df.corr()
    target_correlations = corr_matrix['POWER_ON'].abs().sort_values(ascending=False)
    
    print("Correlations with POWER_ON (absolute values):")
    for feature, corr in target_correlations.items():
        if feature != 'POWER_ON':
            print(f"  {feature}: {corr:.6f}")
    
    # Check for suspiciously high correlations
    suspicious_features = []
    for feature, corr in target_correlations.items():
        if feature != 'POWER_ON' and corr >= 0.999:
            suspicious_features.append((feature, corr))
            print(f"\n⚠️  SUSPICIOUS: {feature} has correlation {corr:.6f} with POWER_ON")
    
    # 2. Check for identical columns
    print("\n" + "="*60)
    print("2. CHECKING FOR IDENTICAL COLUMNS")
    print("="*60)
    
    identical_features = []
    for i, col1 in enumerate(features.columns):
        for j, col2 in enumerate(features.columns):
            if i < j:  # Avoid duplicate comparisons
                if features[col1].equals(features[col2]):
                    identical_features.append((col1, col2))
                    print(f"⚠️  IDENTICAL: {col1} and {col2} are identical")
    
    # Check if any feature is identical to target
    target_identical = []
    for col in features.columns:
        if features[col].equals(target):
            target_identical.append(col)
            print(f"🚨 CRITICAL: {col} is identical to POWER_ON target!")
    
    # 3. Check for constant features
    print("\n" + "="*60)
    print("3. CHECKING FOR CONSTANT FEATURES")
    print("="*60)
    
    constant_features = []
    for col in features.columns:
        if features[col].nunique() == 1:
            constant_features.append(col)
            print(f"⚠️  CONSTANT: {col} has only one unique value: {features[col].iloc[0]}")
    
    # 4. Check for features that are perfect predictors
    print("\n" + "="*60)
    print("4. CHECKING FOR PERFECT PREDICTORS")
    print("="*60)
    
    perfect_predictors = []
    for col in features.columns:
        # Check if feature perfectly separates classes
        unique_values = features[col].unique()
        if len(unique_values) == 2:  # Binary feature
            # Check if each value corresponds to only one class
            value_class_mapping = {}
            for val in unique_values:
                mask = features[col] == val
                classes = target[mask].unique()
                if len(classes) == 1:
                    value_class_mapping[val] = classes[0]
            
            if len(value_class_mapping) == 2:
                perfect_predictors.append(col)
                print(f"🚨 PERFECT PREDICTOR: {col} perfectly separates classes!")
                print(f"   Value {list(value_class_mapping.keys())[0]} -> Class {list(value_class_mapping.values())[0]}")
                print(f"   Value {list(value_class_mapping.keys())[1]} -> Class {list(value_class_mapping.values())[1]}")
    
    # 5. Check for features that are transformations of target
    print("\n" + "="*60)
    print("5. CHECKING FOR TARGET TRANSFORMATIONS")
    print("="*60)
    
    target_transforms = []
    for col in features.columns:
        # Check if feature is a simple transformation of target
        if features[col].equals(1 - target):  # Inverse
            target_transforms.append((col, "inverse"))
            print(f"🚨 TARGET TRANSFORM: {col} is inverse of POWER_ON")
        elif features[col].equals(target * 2):  # Scaled
            target_transforms.append((col, "scaled"))
            print(f"🚨 TARGET TRANSFORM: {col} is scaled POWER_ON")
        elif features[col].equals(target + 1):  # Shifted
            target_transforms.append((col, "shifted"))
            print(f"🚨 TARGET TRANSFORM: {col} is shifted POWER_ON")
    
    # 6. Statistical analysis
    print("\n" + "="*60)
    print("6. STATISTICAL ANALYSIS")
    print("="*60)
    
    print("Feature statistics:")
    print(features.describe())
    
    # 7. Visual analysis
    print("\n" + "="*60)
    print("7. GENERATING VISUAL ANALYSIS")
    print("="*60)
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f')
    plt.title('Correlation Matrix - All Variables')
    plt.tight_layout()
    plt.savefig('results/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved correlation heatmap to results/correlation_heatmap.png")
    
    # Feature vs target scatter plots for top correlations
    top_features = target_correlations.head(6).index[1:]  # Exclude POWER_ON itself
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        if i < len(axes):
            axes[i].scatter(features[feature], target, alpha=0.6)
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('POWER_ON')
            axes[i].set_title(f'{feature} vs POWER_ON\n(corr: {target_correlations[feature]:.3f})')
    
    plt.tight_layout()
    plt.savefig('results/feature_target_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved feature-target scatter plots to results/feature_target_scatter.png")
    
    # 8. Summary report
    print("\n" + "="*80)
    print("DATA LEAKAGE SUMMARY")
    print("="*80)
    
    issues_found = []
    
    if suspicious_features:
        issues_found.append(f"High correlations (≥0.999): {len(suspicious_features)} features")
    
    if identical_features:
        issues_found.append(f"Identical feature pairs: {len(identical_features)} pairs")
    
    if target_identical:
        issues_found.append(f"Features identical to target: {len(target_identical)} features")
    
    if constant_features:
        issues_found.append(f"Constant features: {len(constant_features)} features")
    
    if perfect_predictors:
        issues_found.append(f"Perfect predictors: {len(perfect_predictors)} features")
    
    if target_transforms:
        issues_found.append(f"Target transformations: {len(target_transforms)} features")
    
    if issues_found:
        print("🚨 DATA LEAKAGE DETECTED!")
        print("\nIssues found:")
        for issue in issues_found:
            print(f"  - {issue}")
        
        print("\n🚨 RECOMMENDATION: DO NOT SUBMIT THIS PROJECT")
        print("The perfect accuracy (100%) combined with these issues indicates")
        print("severe data leakage. The model is essentially cheating.")
        
        # Create safe dataset
        print("\nCreating safe dataset by removing problematic features...")
        safe_features = features.copy()
        
        # Remove problematic features
        features_to_remove = set()
        features_to_remove.update([f[0] for f in suspicious_features])
        features_to_remove.update([f[0] for f in identical_features])
        features_to_remove.update(target_identical)
        features_to_remove.update(constant_features)
        features_to_remove.update(perfect_predictors)
        features_to_remove.update([f[0] for f in target_transforms])
        
        safe_features = safe_features.drop(columns=list(features_to_remove))
        
        # Create safe dataset
        safe_df = pd.concat([target, safe_features], axis=1)
        safe_df.to_csv('results/safe_dataframe.csv', index=False)
        
        print(f"Removed {len(features_to_remove)} problematic features:")
        for feature in features_to_remove:
            print(f"  - {feature}")
        
        print(f"\nSafe dataset saved to results/safe_dataframe.csv")
        print(f"Safe dataset shape: {safe_df.shape}")
        
    else:
        print("✅ No obvious data leakage detected")
        print("However, perfect accuracy is still suspicious and requires investigation")
    
    return {
        'suspicious_features': suspicious_features,
        'identical_features': identical_features,
        'target_identical': target_identical,
        'constant_features': constant_features,
        'perfect_predictors': perfect_predictors,
        'target_transforms': target_transforms,
        'issues_found': issues_found
    }

if __name__ == "__main__":
    results = analyze_data_leakage()
