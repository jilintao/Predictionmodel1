"""
Script to check what columns are required by the trained model
"""

import joblib
import pandas as pd
import sys
import io

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Load the model to inspect its requirements
MODEL_FILE = 'trial_fund_model.joblib'

try:
    print("="*60)
    print("Loading model to check required columns...")
    print("="*60)
    
    pipeline = joblib.load(MODEL_FILE)
    
    print("\n[OK] Model loaded successfully!\n")
    
    # Check if it's a CompletePipeline
    if hasattr(pipeline, 'feature_columns'):
        feature_columns = pipeline.feature_columns
        print("="*60)
        print("REQUIRED FEATURE COLUMNS:")
        print("="*60)
        print(f"\nTotal columns needed: {len(feature_columns)}\n")
        
        # Save to file first
        with open('required_columns.txt', 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("REQUIRED COLUMNS FOR PREDICTION\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total columns needed: {len(feature_columns)}\n\n")
            f.write("LIST OF REQUIRED COLUMNS:\n")
            f.write("-"*60 + "\n")
            for i, col in enumerate(feature_columns, 1):
                f.write(f"{i:3d}. {col}\n")
                try:
                    print(f"{i:3d}. {col}")
                except:
                    print(f"{i:3d}. [Column name contains special characters - see required_columns.txt]")
            
            f.write("\n" + "="*60 + "\n")
            f.write("IMPORTANT NOTES:\n")
            f.write("="*60 + "\n")
            f.write("1. Your Excel file MUST have 'user_id' column\n")
            f.write("2. All the above columns should be present in your Excel file\n")
            f.write("3. Missing columns will be automatically filled with 0\n")
            f.write("4. Missing values will be handled automatically:\n")
            f.write("   - Numerical columns: filled with 0\n")
            f.write("   - VIP_level, spot_level, asset_level: filled with 0\n")
            f.write("   - user_tag: filled with '未持仓用户'\n")
            f.write("5. Target encoding will be applied automatically\n")
            f.write("\n" + "="*60 + "\n")
            f.write("COLUMNS THAT WILL BE TARGET ENCODED:\n")
            f.write("="*60 + "\n")
            if hasattr(pipeline, 'target_encode_columns'):
                for col in pipeline.target_encode_columns:
                    f.write(f"  - {col}\n")
        
        print("\n" + "="*60)
        print("SUMMARY:")
        print("="*60)
        print(f"Total columns needed: {len(feature_columns)}")
        print("Plus 'user_id' column (required)")
        print("\n[INFO] Full column list saved to: required_columns.txt")
        print("\nNote: Missing columns will be automatically filled with 0")
        
    else:
        print("Could not find feature_columns in the model.")
        print("Model type:", type(pipeline))
        if hasattr(pipeline, '__dict__'):
            print("Model attributes:", list(pipeline.__dict__.keys()))
    
except FileNotFoundError:
    print(f"ERROR: Model file '{MODEL_FILE}' not found!")
    print("Please ensure the model file exists in the current directory.")
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
