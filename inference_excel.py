"""
Excel-based Inference Script for Trial Fund Usage Prediction

This script loads a saved CompletePipeline and makes predictions on new Excel files.
The pipeline automatically handles all preprocessing (imputation, encoding) for dirty data.
The output is saved as an Excel file with user_id and 预测结果 columns.
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os
from pathlib import Path


def predict_excel(input_excel_path, model_path='trial_fund_model.joblib', output_excel_path=None):
    """
    Make predictions on an Excel file using the saved CompletePipeline.
    The pipeline automatically handles all preprocessing for dirty data.
    
    Parameters:
    -----------
    input_excel_path : str
        Path to the input Excel file (can have missing values, extra columns, etc.)
    model_path : str
        Path to the saved .joblib model file (default: 'trial_fund_model.joblib')
    output_excel_path : str, optional
        Path to save the output Excel file. If None, will be auto-generated.
    
    Returns:
    --------
    pd.DataFrame : DataFrame with user_id and 预测结果 columns
    """
    # Check if files exist
    if not os.path.exists(input_excel_path):
        raise FileNotFoundError(f"Input Excel file not found: {input_excel_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first.")
    
    print(f"Loading pipeline from: {model_path}")
    pipeline = joblib.load(model_path)
    
    print(f"Loading data from: {input_excel_path}")
    df = pd.read_excel(input_excel_path)
    
    print(f"Input data shape: {df.shape}")
    print(f"Input columns: {df.columns.tolist()}")
    
    # Extract user_id if it exists (keep it separate)
    if 'user_id' in df.columns:
        user_ids = df['user_id'].copy()
    else:
        # If no user_id, create sequential IDs
        user_ids = pd.Series(range(len(df)), name='user_id')
        print("Warning: 'user_id' column not found. Using sequential IDs.")
    
    # Remove target column if present (for inference, we don't need it)
    target_columns = ['总体验金使用', 'total_trial_fund_usage']
    features_df = df.drop(columns=target_columns, errors='ignore')
    
    # Remove user_id from features (pipeline doesn't need it)
    if 'user_id' in features_df.columns:
        features_df = features_df.drop(columns=['user_id'])
    
    print(f"\nProcessing {len(features_df)} records...")
    print("Pipeline will automatically:")
    print("  - Fill missing numerical values with 0")
    print("  - Fill missing categorical values (VIP_level, spot_level, asset_level) with 0")
    print("  - Fill missing user_tag with '未持仓用户'")
    print("  - Apply target encoding to categorical features")
    print("  - Handle extra/missing columns automatically")
    
    # The pipeline handles everything automatically!
    # It will:
    # 1. Apply all imputation rules
    # 2. Apply target encoding
    # 3. Handle missing/extra columns
    # 4. Make predictions with custom threshold (0.6)
    
    try:
        # Get predictions using the pipeline
        # The pipeline.predict() method uses the custom threshold (0.6)
        print("\nMaking predictions...")
        predictions = pipeline.predict(features_df)
        
        # Create output DataFrame
        results_df = pd.DataFrame({
            'user_id': user_ids,
            '预测结果': predictions
        })
        
        # Save to Excel
        if output_excel_path is None:
            # Auto-generate output filename
            input_path = Path(input_excel_path)
            output_excel_path = input_path.parent / f"{input_path.stem}_predictions.xlsx"
        
        print(f"\nSaving predictions to: {output_excel_path}")
        results_df.to_excel(output_excel_path, index=False)
        
        print(f"\n{'='*50}")
        print(f"Prediction Summary:")
        print(f"{'='*50}")
        print(f"Total records: {len(results_df)}")
        print(f"Predicted '1' (Yes): {predictions.sum()}")
        print(f"Predicted '0' (No): {(predictions == 0).sum()}")
        print(f"Prediction rate (Yes): {predictions.mean():.2%}")
        print(f"{'='*50}")
        
        return results_df
        
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure the input Excel has the same column structure as training data")
        print("2. Check that the pipeline was saved correctly during training")
        print("3. Verify all required columns are present (pipeline will handle missing ones)")
        raise


def main():
    """
    Main function for command-line usage.
    Usage: python inference_excel.py <input_excel_path> [model_path] [output_excel_path]
    """
    if len(sys.argv) < 2:
        print("Usage: python inference_excel.py <input_excel_path> [model_path] [output_excel_path]")
        print("\nExample:")
        print("  python inference_excel.py new_data.xlsx")
        print("  python inference_excel.py new_data.xlsx trial_fund_model.joblib output.xlsx")
        print("\nNote: The pipeline automatically handles dirty data (missing values, extra columns, etc.)")
        sys.exit(1)
    
    input_excel_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'trial_fund_model.joblib'
    output_excel_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        results = predict_excel(input_excel_path, model_path, output_excel_path)
        print("\n✓ Prediction completed successfully!")
        print(f"✓ Results saved to: {output_excel_path}")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
