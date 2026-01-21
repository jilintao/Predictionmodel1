"""
Flask Web Application for Trial Fund Usage Prediction
Upload Excel file and download prediction results
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from flask import Flask, request, render_template, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
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

# Initialize Flask app
# Explicitly set template folder to ensure templates are found
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

# Model file path
MODEL_FILE = 'trial_fund_model.joblib'

# Load model at startup
print("Loading model...")
try:
    pipeline = joblib.load(MODEL_FILE)
    print(f"✓ Model loaded successfully from {MODEL_FILE}")
except FileNotFoundError:
    print(f"✗ ERROR: Model file '{MODEL_FILE}' not found!")
    print("Please ensure the model file exists in the current directory.")
    pipeline = None
except Exception as e:
    print(f"✗ ERROR loading model: {str(e)}")
    pipeline = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and return predictions."""
    if pipeline is None:
        flash('Model not loaded. Please check that trial_fund_model.joblib exists.', 'error')
        return redirect(url_for('index'))
    
    # Check if file was uploaded
    if 'file' not in request.files:
        flash('No file uploaded. Please select an Excel file.', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        flash('No file selected. Please choose an Excel file.', 'error')
        return redirect(url_for('index'))
    
    # Check if file extension is allowed
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload an Excel file (.xlsx or .xls).', 'error')
        return redirect(url_for('index'))
    
    try:
        # Read the uploaded Excel file
        df = pd.read_excel(file)
        
        if df.empty:
            flash('The uploaded file is empty.', 'error')
            return redirect(url_for('index'))
        
        # Extract user_id if it exists
        if 'user_id' in df.columns:
            user_ids = df['user_id'].copy()
        else:
            user_ids = pd.Series(range(len(df)), name='user_id')
        
        # Remove target column if present (for inference, we don't need it)
        target_columns = ['总体验金使用', 'total_trial_fund_usage']
        features_df = df.drop(columns=target_columns, errors='ignore')
        
        # Remove user_id from features (pipeline doesn't need it)
        if 'user_id' in features_df.columns:
            features_df = features_df.drop(columns=['user_id'])
        
        # Make predictions using the pipeline
        # The pipeline automatically handles all preprocessing!
        predictions = pipeline.predict(features_df)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'user_id': user_ids,
            '预测结果': predictions
        })
        
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False, sheet_name='Predictions')
        output.seek(0)
        
        # Return file for download
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='prediction_results.xlsx'
        )
        
    except pd.errors.EmptyDataError:
        flash('The uploaded file is empty or corrupted.', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        import traceback
        traceback.print_exc()
        return redirect(url_for('index'))


if __name__ == '__main__':
    if pipeline is None:
        print("\n" + "="*60)
        print("WARNING: Model not loaded. The application will start")
        print("but predictions will not work until the model is available.")
        print("="*60 + "\n")
    
    print("\n" + "="*60)
    print("Flask Application Starting...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
