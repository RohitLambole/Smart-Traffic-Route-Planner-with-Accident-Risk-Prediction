"""
train_model_v4.py - Full Data Training + Submit Best
Flipkart Gridlock Hackathon 2.0

Strategy: Since test is Day 49 at specific timestamps, and we validated
that XGBoost with V2 features works best, this version:
1. Trains on ALL training data (Day 48 + Day 49) - no holdout
2. Uses the best hyperparams from V3
3. Generates the final submission

This gives the model more data to learn from, especially Day 49 patterns
that are most relevant to the test set.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model_v2 import load_and_engineer_v2
from feature_engineering import TARGET_COLUMN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'submissions')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


def train_v4():
    print("=" * 60)
    print("GRIDLOCK HACKATHON 2.0 - V4 (FULL DATA TRAINING)")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading and engineering features (V2)...")
    train_df, test_df, features, geo_stats = load_and_engineer_v2()

    # Use ALL training data (no validation holdout)
    X_train = train_df[features]
    y_train = train_df[TARGET_COLUMN]
    print(f"  Training on ALL {len(X_train)} rows (Day 48 + Day 49)")

    # Also keep validation split for reporting (won't affect training)
    val_mask = train_df['day'] == 49
    X_val = train_df.loc[val_mask, features]
    y_val = train_df.loc[val_mask, TARGET_COLUMN]

    # Train XGBoost on full data
    print("\n[2/3] Training XGBoost on full dataset...")
    print("  (Using best params from V3, n_estimators=386 from early stopping)")

    xgb_model = xgb.XGBRegressor(
        n_estimators=400,  # Slightly above V3 early stopping point
        max_depth=8,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.05,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    t = time.time()
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_time = time.time() - t
    print(f"  Training time: {xgb_time:.1f}s")

    # Check performance on Day 49 portion (for reference only)
    val_pred = np.clip(xgb_model.predict(X_val), 0, 1)
    val_r2 = r2_score(y_val, val_pred)
    print(f"  Day 49 R2 (trained on all data, including Day 49): {val_r2:.6f}")
    print(f"  (This is optimistic since Day 49 is in training set)")

    # Also train LightGBM on full data for ensemble option
    print("\n  Training LightGBM on full dataset...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200,  # Slightly above V3 early stopping point (189)
        max_depth=-1,
        learning_rate=0.02,
        num_leaves=255,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)

    # Generate submissions
    print("\n[3/3] Generating submissions...")
    X_test = test_df[features]

    xgb_test = np.clip(xgb_model.predict(X_test), 0, 1)
    lgb_test = np.clip(lgb_model.predict(X_test), 0, 1)

    # Sub A: Pure XGBoost (best from V3)
    sub_a = pd.DataFrame({'Index': test_df['Index'], 'demand': xgb_test})
    path_a = os.path.join(SUBMISSIONS_DIR, 'submission_v4_xgb.csv')
    sub_a.to_csv(path_a, index=False)
    print(f"\n  [A] XGBoost only:")
    print(f"      Saved: {path_a}")
    print(f"      Stats: min={xgb_test.min():.6f}, max={xgb_test.max():.6f}, mean={xgb_test.mean():.6f}")

    # Sub B: Average ensemble
    avg_pred = (xgb_test + lgb_test) / 2
    avg_pred = np.clip(avg_pred, 0, 1)
    sub_b = pd.DataFrame({'Index': test_df['Index'], 'demand': avg_pred})
    path_b = os.path.join(SUBMISSIONS_DIR, 'submission_v4_ensemble.csv')
    sub_b.to_csv(path_b, index=False)
    print(f"\n  [B] Ensemble (XGB+LGB avg):")
    print(f"      Saved: {path_b}")
    print(f"      Stats: min={avg_pred.min():.6f}, max={avg_pred.max():.6f}, mean={avg_pred.mean():.6f}")

    # Verify format
    sample = pd.read_csv(os.path.join(DATASET_DIR, 'sample_submission.csv'))
    for name, sub in [('A', sub_a), ('B', sub_b)]:
        assert list(sub.columns) == list(sample.columns), f"Sub {name}: Column mismatch!"
        assert len(sub) == 41778, f"Sub {name}: Row count mismatch!"
    print(f"\n  Format check: ALL PASSED")

    # Save models
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_model_v4.pkl'))
    joblib.dump(lgb_model, os.path.join(MODELS_DIR, 'lgb_model_v4.pkl'))
    joblib.dump({
        'features': features, 'geo_stats': geo_stats,
        'use_log_transform': False, 'trained_on': 'all_data'
    }, os.path.join(MODELS_DIR, 'model_meta_v4.pkl'))

    print("\n" + "=" * 60)
    print("V4 COMPLETE!")
    print("=" * 60)
    print(f"\nSubmission files ready:")
    print(f"  1. {path_a} (XGBoost, recommended first upload)")
    print(f"  2. {path_b} (Ensemble, try if XGB doesn't score well)")
    print(f"\nUpload to HackerEarth and check your score!")


if __name__ == '__main__':
    train_v4()
