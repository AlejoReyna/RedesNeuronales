"""
Data preprocessing module for Horno Prediction Project.
Handles data loading, cleaning, scaling, and train-test splitting.
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles all data preprocessing tasks including loading, cleaning,
    scaling, and splitting data for the CNN model.
    """
    
    def __init__(self, data_path='data/Variables_Horno_REAL.csv', train_ratio=0.7, top_k=None, corr_threshold=None, target_clip_quantile=None):
        """
        Initialize the preprocessor.
        
        Args:
            data_path (str): Path to the CSV file
            train_ratio (float): Ratio of data to use for training (default: 0.7)
        """
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_name = None
        self.top_k = top_k
        self.corr_threshold = corr_threshold
        self.target_clip_quantile = target_clip_quantile
        self.target_clip_threshold = None
        
    def load_data(self):
        """
        Load data from CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        logger.info(f"Loading data from {self.data_path}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def clean_data(self, df):
        """
        Clean the data: handle missing values, remove duplicates, etc.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("Starting data cleaning process")
        
        initial_shape = df.shape
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
            
            # Impute numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    median_value = df[col].median()
                    df.loc[:, col] = df[col].fillna(median_value)
                    logger.info(f"Imputed {col} with median: {median_value}")
            
            # Impute categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    mode_value = df[col].mode()[0]
                    df.loc[:, col] = df[col].fillna(mode_value)
                    logger.info(f"Imputed {col} with mode: {mode_value}")
        else:
            logger.info("No missing values found")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove non-numeric columns except target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = df[numeric_cols]
        
        logger.info(f"Data cleaning completed. Shape change: {initial_shape} -> {df.shape}")
        
        return df

    
    def prepare_features_target(self, df):
        """
        Separate features and target variable.
        Assumes POWER ON is the first column (column A / index 0).
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (X, y) features and target
        """
        logger.info("Preparing features and target variable")
        
        # Target is the first column (POWER ON - column A)
        self.target_name = df.columns[0]
        logger.info(f"Target variable: {self.target_name}")
        
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
        
        self.feature_names = df.columns[1:].tolist()
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Number of features: {len(self.feature_names)}")
        logger.info(f"Target distribution - Mean: {y.mean():.4f}, Std: {y.std():.4f}")
        logger.info(f"Target range: [{y.min():.4f}, {y.max():.4f}]")
        
        return X, y
    
    def split_data(self, X, y):
        """
        Split data into train and test sets using temporal split.
        First 70% for training, last 30% for testing.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data with train ratio: {self.train_ratio}")
        
        split_index = int(len(X) * self.train_ratio)
        
        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"Train target mean: {y_train.mean():.4f}")
        logger.info(f"Test target mean: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test

    def clip_target(self, y_train, y_test):
        quantile = float(self.target_clip_quantile)
        if quantile <= 0 or quantile >= 1:
            logger.warning("target_clip_quantile should be between 0 and 1. Skipping clipping.")
            return y_train, y_test

        threshold = np.quantile(y_train, quantile)
        y_train_clipped = np.minimum(y_train, threshold)
        y_test_clipped = np.minimum(y_test, threshold)
        self.target_clip_threshold = threshold

        logger.info(f"Clipped target at quantile {quantile:.3f} (threshold: {threshold:.4f})")
        clipped_count = np.sum(y_train != y_train_clipped)
        logger.info(f"Training samples clipped: {clipped_count}")

        return y_train_clipped, y_test_clipped

    def select_features_by_correlation(self, X_train, y_train, X_test):
        """
        Optionally select features based on correlation with the target on training data.

        Args:
            X_train (np.ndarray): Training features (unscaled)
            y_train (np.ndarray): Training target
            X_test (np.ndarray): Test features (unscaled)

        Returns:
            tuple: (X_train_sel, X_test_sel)
        """
        if self.top_k is None and (self.corr_threshold is None):
            return X_train, X_test

        logger.info("Applying feature selection based on correlation with the target (computed on training set)")
        # Build DataFrame to compute correlations
        train_df = pd.DataFrame(X_train, columns=self.feature_names)
        target_series = pd.Series(y_train, name=self.target_name)

        # Pearson correlation
        corr_with_target = train_df.corrwith(target_series)

        # Determine selected features
        if self.top_k is not None:
            selected = corr_with_target.abs().sort_values(ascending=False).head(int(self.top_k)).index.tolist()
        else:
            threshold = float(self.corr_threshold)
            corr_with_target_abs = corr_with_target.abs()
            selected = corr_with_target_abs[corr_with_target_abs >= threshold].index.tolist()

        if len(selected) == 0:
            logger.warning("No features met the selection criteria; keeping all features.")
            return X_train, X_test

        logger.info(f"Selected {len(selected)} features out of {len(self.feature_names)}")
        self.feature_names = selected

        # Column indices to keep
        name_to_idx = {name: idx for idx, name in enumerate(list(train_df.columns))}
        keep_indices = [name_to_idx[name] for name in selected]

        X_train_sel = X_train[:, keep_indices]
        X_test_sel = X_test[:, keep_indices]

        logger.info(f"Feature selection shapes: train {X_train_sel.shape}, test {X_test_sel.shape}")
        return X_train_sel, X_test_sel
    
    def scale_features(self, X_train, X_test, fit=True, scaler_path='models/scaler.pkl'):
        """
        Scale features using StandardScaler.
        
        Args:
            X_train (np.ndarray): Training features
            X_test (np.ndarray): Test features
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled)
        """
        if fit:
            logger.info("Fitting StandardScaler on training data")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            if self.scaler is None or not hasattr(self.scaler, 'mean_'):
                logger.info("Loading existing scaler for transformation")
                self.load_scaler(scaler_path)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Feature scaling completed")
        
        return X_train_scaled, X_test_scaled
    
    def save_scaler(self, output_path='models/scaler.pkl'):
        """
        Save the fitted scaler to disk.
        
        Args:
            output_path (str): Path to save the scaler
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self.scaler, output_path)
        logger.info(f"Scaler saved to {output_path}")
    
    def load_scaler(self, scaler_path='models/scaler.pkl'):
        """
        Load a saved scaler from disk.
        
        Args:
            scaler_path (str): Path to the saved scaler
        """
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")
    
    def preprocess(self, fit_scaler=True):
        """
        Execute the full preprocessing pipeline.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info("=" * 60)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        
        # Load data
        df = self.load_data()
        
        # Clean data
        df = self.clean_data(df)
        
        # Prepare features and target
        X, y = self.prepare_features_target(df)

        # Split data (before feature selection to avoid leakage)
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        if self.target_clip_quantile is not None:
            y_train, y_test = self.clip_target(y_train, y_test)

        # Optional feature selection on training set statistics
        X_train, X_test = self.select_features_by_correlation(X_train, y_train, X_test)
        
        scaler_path = 'models/scaler.pkl'
        X_train, X_test = self.scale_features(X_train, X_test, fit=fit_scaler, scaler_path=scaler_path)
        
        if fit_scaler:
            self.save_scaler(scaler_path)
        
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return X_train, X_test, y_train, y_test


def main():
    """
    Main function for standalone execution.
    """
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    
    print("\nPreprocessing Summary:")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Target variable: {preprocessor.target_name}")
    print(f"Number of features: {len(preprocessor.feature_names)}")


if __name__ == '__main__':
    main()

