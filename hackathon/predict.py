"""
predict.py - Generate Submission CSV for HackerEarth
Flipkart Gridlock Hackathon 2.0

Loads the trained best model, applies feature engineering to test data,
and generates the submission file.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Add hackathon dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import load_and_engineer, FEATURE_COLUMNS


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'submissions')
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


# ============================================================
# GENERATE SUBMISSION
# ============================================================

def generate_submission(version='v1'):
    """
    Full pipeline: Load model + test data -> predict -> save submission CSV.
    
    Args:
        version: Version string for the submission filename (e.g., 'v1', 'v2')
    
    Returns:
        DataFrame: The submission DataFrame
    """
    print("=" * 60)
    print("GENERATING SUBMISSION")
    print("=" * 60)
    
    # Step 1: Load model and metadata
    print("\n[1/4] Loading trained model...")
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    meta_path = os.path.join(MODELS_DIR, 'model_meta.pkl')
    
    if not os.path.exists(model_path):
        print("ERROR: No trained model found! Run train_model.py first.")
        return None
    
    model = joblib.load(model_path)
    meta = joblib.load(meta_path)
    
    print(f"  Model: {meta['model_name']}")
    print(f"  Features: {len(meta['features'])} columns")
    
    # Step 2: Load and engineer features
    print("\n[2/4] Loading and engineering features...")
    train_df, test_df, features, geo_stats = load_and_engineer()
    
    # Step 3: Predict
    print("\n[3/4] Predicting demand for test set...")
    X_test = test_df[features]
    predictions = model.predict(X_test)
    
    # Clip to valid range (demand is 0-1)
    predictions = np.clip(predictions, 0, 1)
    
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Min: {predictions.min():.6f}")
    print(f"  Max: {predictions.max():.6f}")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Median: {np.median(predictions):.6f}")
    
    # Step 4: Create submission DataFrame
    print("\n[4/4] Creating submission file...")
    submission = pd.DataFrame({
        'Index': test_df['Index'],
        'demand': predictions
    })
    
    # Verify format matches sample_submission.csv
    sample_path = os.path.join(BASE_DIR, '..', 'dataset', 'sample_submission.csv')
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        print(f"\n  Sample submission format: {list(sample.columns)}")
        print(f"  Our submission format:    {list(submission.columns)}")
        assert list(submission.columns) == list(sample.columns), "Column mismatch!"
        print("  Format check: PASSED")
    
    # Save
    output_path = os.path.join(SUBMISSIONS_DIR, f'submission_{version}.csv')
    submission.to_csv(output_path, index=False)
    print(f"\n  Submission saved to: {output_path}")
    print(f"  Total rows: {len(submission)}")
    
    # Quick sanity check
    print("\n  Submission preview:")
    print(submission.head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("SUBMISSION READY!")
    print(f"Upload '{output_path}' to HackerEarth")
    print("=" * 60)
    
    return submission


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate hackathon submission')
    parser.add_argument('--version', '-v', default='v1', help='Submission version (default: v1)')
    args = parser.parse_args()
    
    generate_submission(version=args.version)
