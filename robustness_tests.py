#!/usr/bin/env python3
"""
Robustness Testing Script for HornoPrediction Project
This script performs alternative validation tests to check model robustness.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import tensorflow as tf
from tensorflow import keras
import joblib
import warnings
warnings.filterwarnings('ignore')

def analyze_class_distribution():
    """Analyze class distribution and balance"""
    print("=" * 80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv('data/Variables_Horno.csv')
    target = df['POWER_ON']
    
    # Class distribution
    class_counts = target.value_counts().sort_index()
    class_proportions = target.value_counts(normalize=True).sort_index()
    
    print(f"Total samples: {len(target)}")
    print(f"Class 0 (POWER_OFF): {class_counts[0]} samples ({class_proportions[0]:.1%})")
    print(f"Class 1 (POWER_ON): {class_counts[1]} samples ({class_proportions[1]:.1%})")
    
    # Balance ratio
    balance_ratio = min(class_counts) / max(class_counts)
    print(f"Balance ratio: {balance_ratio:.3f}")
    
    if balance_ratio < 0.3:
        print("⚠️  SEVERELY IMBALANCED: Balance ratio < 0.3")
    elif balance_ratio < 0.5:
        print("⚠️  IMBALANCED: Balance ratio < 0.5")
    else:
        print("✅ RELATIVELY BALANCED: Balance ratio ≥ 0.5")
    
    # Visualize distribution
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    class_counts.plot(kind='bar', color=['red', 'green'])
    plt.title('Class Distribution (Counts)')
    plt.xlabel('POWER_ON')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    plt.subplot(1, 2, 2)
    class_proportions.plot(kind='bar', color=['red', 'green'])
    plt.title('Class Distribution (Proportions)')
    plt.xlabel('POWER_ON')
    plt.ylabel('Proportion')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('results/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved class distribution plot to results/class_distribution.png")
    
    return {
        'class_counts': class_counts,
        'class_proportions': class_proportions,
        'balance_ratio': balance_ratio,
        'is_balanced': balance_ratio >= 0.5
    }

def test_shuffled_split():
    """Test model with shuffled (random) train/test split"""
    print("\n" + "=" * 80)
    print("SHUFFLED SPLIT TEST")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv('data/Variables_Horno.csv')
    target = df['POWER_ON']
    features = df.drop('POWER_ON', axis=1)
    
    # Shuffled split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42, stratify=target
    )
    
    print(f"Shuffled split - Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build simple CNN model (smaller for faster training)
    model = keras.Sequential([
        keras.layers.Reshape((X_train_scaled.shape[1], 1), input_shape=(X_train_scaled.shape[1],)),
        keras.layers.Conv1D(16, 3, activation='relu'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(32, 3, activation='relu'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=['accuracy', 'precision', 'recall'])
    
    # Train with early stopping
    early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nShuffled Split Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss (Shuffled Split)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy (Shuffled Split)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['precision'], label='Train Precision')
    plt.plot(history.history['val_precision'], label='Val Precision')
    plt.title('Model Precision (Shuffled Split)')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/shuffled_split_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved shuffled split training curves to results/shuffled_split_curves.png")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'history': history.history
    }

def test_kfold_cv():
    """Test model with K-Fold Cross Validation"""
    print("\n" + "=" * 80)
    print("K-FOLD CROSS VALIDATION TEST")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv('data/Variables_Horno.csv')
    target = df['POWER_ON']
    features = df.drop('POWER_ON', axis=1)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-Fold CV
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_scores = []
    fold_histories = []
    
    print("Running 5-Fold Cross Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(features_scaled)):
        print(f"Fold {fold + 1}/5...")
        
        X_train_fold = features_scaled[train_idx]
        X_val_fold = features_scaled[val_idx]
        y_train_fold = target.iloc[train_idx]
        y_val_fold = target.iloc[val_idx]
        
        # Build model
        model = keras.Sequential([
            keras.layers.Reshape((X_train_fold.shape[1], 1), input_shape=(X_train_fold.shape[1],)),
            keras.layers.Conv1D(16, 3, activation='relu'),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(32, 3, activation='relu'),
            keras.layers.MaxPooling1D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', 
                      metrics=['accuracy', 'precision', 'recall'])
        
        # Train
        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=30,
            batch_size=32,
            validation_data=(X_val_fold, y_val_fold),
            verbose=0
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_val_fold, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_val_fold, y_pred)
        precision = precision_score(y_val_fold, y_pred)
        recall = recall_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred)
        
        fold_scores.append({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        fold_histories.append(history.history)
    
    # Calculate statistics
    accuracies = [score['accuracy'] for score in fold_scores]
    precisions = [score['precision'] for score in fold_scores]
    recalls = [score['recall'] for score in fold_scores]
    f1s = [score['f1'] for score in fold_scores]
    
    print(f"\nK-Fold CV Results:")
    print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    
    # Plot CV results
    plt.figure(figsize=(15, 10))
    
    # Box plots
    plt.subplot(2, 3, 1)
    plt.boxplot([accuracies, precisions, recalls, f1s], 
                labels=['Accuracy', 'Precision', 'Recall', 'F1'])
    plt.title('K-Fold CV Score Distribution')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    
    # Individual fold results
    plt.subplot(2, 3, 2)
    plt.plot(range(1, 6), accuracies, 'o-', label='Accuracy')
    plt.plot(range(1, 6), precisions, 's-', label='Precision')
    plt.plot(range(1, 6), recalls, '^-', label='Recall')
    plt.plot(range(1, 6), f1s, 'd-', label='F1')
    plt.title('K-Fold CV Scores by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Training curves for first fold
    plt.subplot(2, 3, 3)
    plt.plot(fold_histories[0]['loss'], label='Train Loss')
    plt.plot(fold_histories[0]['val_loss'], label='Val Loss')
    plt.title('Training Curves (Fold 1)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 3, 4)
    plt.plot(fold_histories[0]['accuracy'], label='Train Accuracy')
    plt.plot(fold_histories[0]['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy Curves (Fold 1)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.plot(fold_histories[0]['precision'], label='Train Precision')
    plt.plot(fold_histories[0]['val_precision'], label='Val Precision')
    plt.title('Precision Curves (Fold 1)')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    plt.plot(fold_histories[0]['recall'], label='Train Recall')
    plt.plot(fold_histories[0]['val_recall'], label='Val Recall')
    plt.title('Recall Curves (Fold 1)')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/kfold_cv_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved K-Fold CV results to results/kfold_cv_results.png")
    
    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_precision': np.mean(precisions),
        'std_precision': np.std(precisions),
        'mean_recall': np.mean(recalls),
        'std_recall': np.std(recalls),
        'mean_f1': np.mean(f1s),
        'std_f1': np.std(f1s),
        'fold_scores': fold_scores
    }

def test_without_top_features():
    """Test model without top correlated features"""
    print("\n" + "=" * 80)
    print("TEST WITHOUT TOP CORRELATED FEATURES")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv('data/Variables_Horno.csv')
    target = df['POWER_ON']
    features = df.drop('POWER_ON', axis=1)
    
    # Calculate correlations
    corr_matrix = df.corr()
    target_correlations = corr_matrix['POWER_ON'].abs().sort_values(ascending=False)
    
    # Remove top 5 most correlated features
    top_features = target_correlations.head(6).index[1:]  # Exclude POWER_ON itself
    features_reduced = features.drop(columns=top_features)
    
    print(f"Removed top correlated features: {list(top_features)}")
    print(f"Original features: {features.shape[1]}")
    print(f"Reduced features: {features_reduced.shape[1]}")
    
    # Temporal split (70/30)
    split_idx = int(0.7 * len(features_reduced))
    X_train = features_reduced.iloc[:split_idx]
    X_test = features_reduced.iloc[split_idx:]
    y_train = target.iloc[:split_idx]
    y_test = target.iloc[split_idx:]
    
    print(f"Temporal split - Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build model
    model = keras.Sequential([
        keras.layers.Reshape((X_train_scaled.shape[1], 1), input_shape=(X_train_scaled.shape[1],)),
        keras.layers.Conv1D(16, 3, activation='relu'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(32, 3, activation='relu'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=['accuracy', 'precision', 'recall'])
    
    # Train
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluate
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nResults without top features:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'removed_features': list(top_features)
    }

def main():
    """Run all robustness tests"""
    print("Starting comprehensive robustness testing...")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run tests
    class_dist = analyze_class_distribution()
    shuffled_results = test_shuffled_split()
    kfold_results = test_kfold_cv()
    reduced_features_results = test_without_top_features()
    
    # Summary
    print("\n" + "=" * 80)
    print("ROBUSTNESS TESTING SUMMARY")
    print("=" * 80)
    
    print(f"Class Distribution:")
    print(f"  Balance ratio: {class_dist['balance_ratio']:.3f}")
    print(f"  Is balanced: {class_dist['is_balanced']}")
    
    print(f"\nShuffled Split Test:")
    print(f"  Accuracy: {shuffled_results['accuracy']:.4f}")
    print(f"  F1-Score: {shuffled_results['f1']:.4f}")
    
    print(f"\nK-Fold CV Test:")
    print(f"  Accuracy: {kfold_results['mean_accuracy']:.4f} ± {kfold_results['std_accuracy']:.4f}")
    print(f"  F1-Score: {kfold_results['mean_f1']:.4f} ± {kfold_results['std_f1']:.4f}")
    
    print(f"\nWithout Top Features Test:")
    print(f"  Accuracy: {reduced_features_results['accuracy']:.4f}")
    print(f"  F1-Score: {reduced_features_results['f1']:.4f}")
    
    # Compare with original results
    print(f"\nComparison with Original Results:")
    print(f"  Original Accuracy: 1.0000 (100%)")
    print(f"  Shuffled Accuracy: {shuffled_results['accuracy']:.4f}")
    print(f"  K-Fold Accuracy: {kfold_results['mean_accuracy']:.4f}")
    print(f"  Reduced Features Accuracy: {reduced_features_results['accuracy']:.4f}")
    
    # Analysis
    accuracy_drop = 1.0 - shuffled_results['accuracy']
    print(f"\nAnalysis:")
    print(f"  Accuracy drop with shuffled split: {accuracy_drop:.4f}")
    
    if accuracy_drop > 0.1:
        print("  🚨 SIGNIFICANT DROP: Model performance drops significantly with shuffled split")
        print("  This suggests the model may be overfitting to temporal patterns")
    elif accuracy_drop > 0.05:
        print("  ⚠️  MODERATE DROP: Model performance drops moderately with shuffled split")
    else:
        print("  ✅ ROBUST: Model maintains good performance with shuffled split")
    
    return {
        'class_distribution': class_dist,
        'shuffled_results': shuffled_results,
        'kfold_results': kfold_results,
        'reduced_features_results': reduced_features_results
    }

if __name__ == "__main__":
    results = main()
