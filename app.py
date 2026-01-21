"""
Streamlit Web Application for Trial Fund Usage Prediction
Upload Excel file and download prediction results
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder
import io

# ============================================================================
# CUSTOM TRANSFORMER CLASS DEFINITIONS
# These classes must be defined here so joblib can unpickle the saved model
# ============================================================================

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
        
        # Ensure feature_names matches the actual number of columns
        # If there's a mismatch, use the actual number of columns from X_processed
        if X_processed.shape[1] != len(feature_names):
            # If mismatch, create generic column names
            feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]
        
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index if hasattr(X, 'index') else range(len(X_processed)))
        
        # Apply target encoding
        for col in self.target_encode_columns:
            if col in X_processed_df.columns and col in self.target_encoders:
                encoder = self.target_encoders[col]
                X_processed_df[col] = encoder.transform(X_processed_df[[col]])
        
        # Ensure all columns are present and in the correct order
        if self.feature_columns:
            # Add missing columns with 0
            for col in self.feature_columns:
                if col not in X_processed_df.columns:
                    X_processed_df[col] = 0
            
            # Select only the columns used in training, in the correct order
            # Handle case where some columns might not exist
            available_cols = [col for col in self.feature_columns if col in X_processed_df.columns]
            missing_cols = [col for col in self.feature_columns if col not in X_processed_df.columns]
            
            if missing_cols:
                # Add missing columns with 0
                for col in missing_cols:
                    X_processed_df[col] = 0
            
            # Reorder to match training order
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

# ============================================================================
# END OF CUSTOM TRANSFORMER CLASS DEFINITIONS
# ============================================================================

# Model file path
MODEL_FILE = 'trial_fund_model.joblib'

# Load model at startup (with caching)
@st.cache_resource
def load_model():
    """Load the model with caching."""
    try:
        import warnings
        import pickle
        import sys
        import pandas as pd
        
        # Suppress warnings during loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Fix for pandas StringDtype compatibility across versions
            # This handles cases where the model was saved with one pandas version
            # and loaded with another (even within 1.x range)
            try:
                # Ensure pandas.core.arrays.string_ module is available
                if hasattr(pd.core.arrays, 'string_'):
                    sys.modules['pandas.core.arrays.string'] = pd.core.arrays.string_
                elif hasattr(pd.core.arrays, 'string'):
                    sys.modules['pandas.core.arrays.string_'] = pd.core.arrays.string
            except:
                pass
            
            # Load model - handle pandas StringDtype compatibility issues
            try:
                # Standard load
                pipeline = joblib.load(MODEL_FILE)
            except (TypeError, ValueError, AttributeError, pickle.UnpicklingError) as e:
                # Handle StringDtype compatibility issues
                error_str = str(e)
                if "StringDtype" in error_str or "dtype" in error_str.lower():
                    # Try to create a compatibility shim for StringDtype
                    try:
                        # Patch StringDtype if it's causing issues
                        if hasattr(pd, 'StringDtype'):
                            original_string_dtype = pd.StringDtype
                            
                            # Create a compatibility wrapper
                            class StringDtypeCompat:
                                def __new__(cls, *args, **kwargs):
                                    # Try to use original StringDtype
                                    try:
                                        return original_string_dtype(*args, **kwargs)
                                    except:
                                        # Fallback: return object dtype
                                        return pd.api.types.pandas_dtype('object')
                            
                            # Temporarily patch
                            pd.StringDtype = StringDtypeCompat
                            
                            try:
                                pipeline = joblib.load(MODEL_FILE)
                            finally:
                                # Restore original
                                pd.StringDtype = original_string_dtype
                        else:
                            # If StringDtype doesn't exist, try loading anyway
                            pipeline = joblib.load(MODEL_FILE)
                    except Exception as e2:
                        # Last resort: provide helpful error
                        raise ValueError(
                            f"Model loading failed due to pandas StringDtype compatibility.\n"
                            f"Original error: {str(e)[:200]}\n"
                            f"Retry error: {str(e2)[:200]}\n\n"
                            f"This usually happens when the model was saved with pandas 1.3+ "
                            f"and you're loading with a different version.\n\n"
                            f"**Solution:**\n"
                            f"1. Ensure your requirements.txt specifies: pandas>=1.5.0,<2.0.0\n"
                            f"2. Use Python 3.11 on Streamlit Cloud (pandas 1.x doesn't officially support Python 3.13)\n"
                            f"3. Or retrain the model with the exact pandas version used in Streamlit Cloud"
                        )
                else:
                    raise
        
        return pipeline, None
    except FileNotFoundError:
        return None, f"Model file '{MODEL_FILE}' not found! Please ensure the model file is uploaded to Streamlit Cloud."
    except Exception as e:
        error_msg = str(e)
        # Provide more helpful error message
        if "StringDtype" in error_msg:
            return None, (
                f"Model compatibility error with pandas StringDtype.\n\n"
                f"This usually happens when the model was saved with a different pandas version.\n\n"
                f"**Solution:**\n"
                f"1. Ensure your requirements.txt specifies: pandas>=1.3.0,<2.0.0\n"
                f"2. Or retrain the model with the pandas version used in Streamlit Cloud\n"
                f"3. Error details: {error_msg[:200]}"
            )
        return None, f"Error loading model: {error_msg[:500]}"


# Page configuration
st.set_page_config(
    page_title="Contract Trial Fund Prediction Tool",
    page_icon="📊",
    layout="centered"
)

# Title
st.title("📊 Contract Trial Fund Prediction Tool")

# Load model
pipeline, model_error = load_model()

if model_error:
    st.error(f"❌ {model_error}")
    st.stop()

if pipeline is None:
    st.error("❌ Model not loaded. Please check that trial_fund_model.joblib exists.")
    st.stop()

st.success("✅ Model loaded successfully!")

# File uploader
st.markdown("---")
st.header("Upload Excel File")

uploaded_file = st.file_uploader(
    "Choose an Excel file (.xlsx or .xls)",
    type=['xlsx', 'xls'],
    help="Upload your Excel file with user data for prediction"
)

if uploaded_file is not None:
    try:
        # Read the uploaded Excel file
        df = pd.read_excel(uploaded_file)
        
        if df.empty:
            st.error("❌ The uploaded file is empty.")
            st.stop()
        
        st.success(f"✅ File loaded successfully! ({len(df)} records)")
        
        # Show file info
        with st.expander("📋 File Information"):
            st.write(f"**Total rows:** {len(df)}")
            st.write(f"**Total columns:** {len(df.columns)}")
            st.write("**Columns in file:**")
            st.write(df.columns.tolist())
        
        # Extract user_id if it exists
        if 'user_id' in df.columns:
            user_ids = df['user_id'].copy()
            st.info("✓ Found user_id column")
        else:
            user_ids = pd.Series(range(len(df)), name='user_id')
            st.warning("⚠ user_id column not found. Using sequential IDs.")
        
        # Remove target column if present (for inference, we don't need it)
        target_columns = ['总体验金使用', 'total_trial_fund_usage']
        features_df = df.drop(columns=target_columns, errors='ignore')
        
        # Remove user_id from features (pipeline doesn't need it)
        if 'user_id' in features_df.columns:
            features_df = features_df.drop(columns=['user_id'])
        
        # Make predictions
        if st.button("🔮 Predict & Generate Results", type="primary", use_container_width=True):
            with st.spinner("Processing predictions..."):
                try:
                    # The pipeline handles all preprocessing automatically!
                    predictions = pipeline.predict(features_df)
                    probabilities = pipeline.predict_proba(features_df)[:, 1]
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame({
                        'user_id': user_ids,
                        '预测结果': predictions
                    })
                    
                    # Display summary
                    st.success("✅ Predictions completed!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", len(results_df))
                    with col2:
                        st.metric("Predicted '1' (Yes)", int(predictions.sum()))
                    with col3:
                        st.metric("Predicted '0' (No)", int((predictions == 0).sum()))
                    
                    # Show preview
                    with st.expander("👀 Preview Results"):
                        st.dataframe(results_df.head(20))
                    
                    # Create Excel file in memory
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        results_df.to_excel(writer, index=False, sheet_name='Predictions')
                    output.seek(0)
                    
                    # Download button
                    st.markdown("---")
                    st.download_button(
                        label="📥 Download Prediction Results",
                        data=output,
                        file_name="prediction_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        type="primary"
                    )
                    
                    st.info("💡 The downloaded file contains two columns: **user_id** and **预测结果**")
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
        
    except pd.errors.EmptyDataError:
        st.error("❌ The uploaded file is empty or corrupted.")
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        import traceback
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

else:
    st.info("👆 Please upload an Excel file to get started.")

# Instructions
with st.expander("ℹ️ Instructions"):
    st.markdown("""
    **How to use:**
    1. Click "Choose an Excel file" to select your data file (.xlsx or .xls)
    2. Review the file information
    3. Click "Predict & Generate Results" to process your data
    4. Download the results Excel file
    
    **Output file contains:**
    - `user_id`: User identifier
    - `预测结果`: Prediction result (0 or 1)
    
    **Note:** The pipeline automatically handles:
    - Missing values (filled with 0 or '未持仓用户')
    - Target encoding for categorical features
    - Extra/missing columns
    """)
