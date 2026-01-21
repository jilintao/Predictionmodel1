"""
Custom Transformers for the Trial Fund Usage Prediction Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder


class NumericalImputer(BaseEstimator, TransformerMixin):
    """Fill NaN values with 0 for numerical columns."""
    
    def __init__(self, fill_value=0):
        self.fill_value = fill_value
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Convert to numeric, coercing errors to NaN, then fill
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(self.fill_value)
        return X


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Fill NaN values with a specified value for categorical columns."""
    
    def __init__(self, fill_value=0):
        self.fill_value = fill_value
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        return X.fillna(self.fill_value)


class StringImputer(BaseEstimator, TransformerMixin):
    """Fill NaN and empty strings with a specified value for string columns."""
    
    def __init__(self, fill_value='未持仓用户'):
        self.fill_value = fill_value
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].fillna(self.fill_value)
            X[col] = X[col].replace('', self.fill_value)
        return X


class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    """Wrapper for TargetEncoder that works with Pipeline."""
    
    def __init__(self):
        self.encoders = {}
        self.columns = None
    
    def fit(self, X, y=None):
        if y is None:
            raise ValueError("TargetEncoderWrapper requires y (target) for fitting")
        
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.columns = X.columns.tolist()
        
        # Fit a TargetEncoder for each column
        for col in self.columns:
            encoder = TargetEncoder()
            encoder.fit(X[[col]], y)
            self.encoders[col] = encoder
        
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Ensure all columns are present
        for col in self.columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in transform data")
        
        # Apply encoding
        X_encoded = X.copy()
        for col in self.columns:
            encoder = self.encoders[col]
            X_encoded[col] = encoder.transform(X[[col]])
        
        return X_encoded


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select specific columns from a DataFrame."""
    
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Add missing columns with 0
        for col in self.columns:
            if col not in X.columns:
                X[col] = 0
        
        # Select only the specified columns in the correct order
        return X[self.columns]


class CompletePipeline:
    """
    Complete pipeline that handles preprocessing and target encoding.
    This is needed because TargetEncoder requires y during fit.
    """
    def __init__(self, preprocessor, target_encode_columns, model, threshold=0.6):
        self.preprocessor = preprocessor
        self.target_encode_columns = target_encode_columns
        self.target_encoders = {}
        self.model = model
        self.threshold = threshold
        self.feature_columns = None
    
    def fit(self, X, y):
        """Fit the pipeline on training data."""
        # Apply initial preprocessing
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names
        feature_names = self._get_feature_names(X)
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        
        # Fit target encoders
        for col in self.target_encode_columns:
            if col in X_processed_df.columns:
                encoder = TargetEncoder()
                X_processed_df[col] = encoder.fit_transform(X_processed_df[[col]], y)
                self.target_encoders[col] = encoder
        
        # Store feature columns for inference
        self.feature_columns = X_processed_df.columns.tolist()
        
        # Fit model
        self.model.fit(X_processed_df, y)
        
        return self
    
    def transform(self, X):
        """Transform data (for inference)."""
        # Apply preprocessing
        X_processed = self.preprocessor.transform(X)
        
        # Get feature names
        feature_names = self._get_feature_names(X)
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        
        # Apply target encoding
        for col in self.target_encode_columns:
            if col in X_processed_df.columns and col in self.target_encoders:
                encoder = self.target_encoders[col]
                X_processed_df[col] = encoder.transform(X_processed_df[[col]])
        
        # Ensure all columns are present
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in X_processed_df.columns:
                    X_processed_df[col] = 0
            X_processed_df = X_processed_df[self.feature_columns]
        
        return X_processed_df
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        X_transformed = self.transform(X)
        return self.model.predict_proba(X_transformed)
    
    def predict(self, X):
        """Get predictions with custom threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba > self.threshold).astype(int)
    
    def _get_feature_names(self, X):
        """Get feature names after preprocessing."""
        feature_names = []
        
        # Get transformers (use transformers_ if fitted, otherwise use transformers)
        if hasattr(self.preprocessor, 'transformers_'):
            transformers_list = self.preprocessor.transformers_
        else:
            transformers_list = self.preprocessor.transformers
        
        processed_cols = set()
        for name, transformer, cols in transformers_list:
            if name != 'remainder':
                processed_cols.update(cols)
                if hasattr(transformer, 'get_feature_names_out'):
                    try:
                        feature_names.extend(transformer.get_feature_names_out(cols))
                    except:
                        feature_names.extend(cols)
                else:
                    feature_names.extend(cols)
        
        # Add remainder columns
        all_cols = set(X.columns)
        remainder_cols = sorted(list(all_cols - processed_cols))
        feature_names.extend(remainder_cols)
        
        return feature_names
