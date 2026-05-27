"""
train_model_v3.py - Best of V1 + V2 (no log transform, expanded features, ensemble)
Flipkart Gridlock Hackathon 2.0
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model_v2 import load_and_engineer_v2, FEATURE_COLUMNS_V2
from feature_engineering import TARGET_COLUMN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'submissions')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


def train_v3():
    print("=" * 60)
    print("GRIDLOCK HACKATHON 2.0 - V3 MODEL TRAINING")
    print("  (V2 features + NO log transform + tuned ensemble)")
    print("=" * 60)

    # Load data with V2 features
    print("\n[1/4] Loading and engineering features (V2)...")
    train_df, test_df, features, geo_stats = load_and_engineer_v2()

    # Time-based split
    print("\n[2/4] Creating time-based validation split...")
    train_mask = train_df['day'] == 48
    val_mask = train_df['day'] == 49

    X_train = train_df.loc[train_mask, features]
    y_train = train_df.loc[train_mask, TARGET_COLUMN]
    X_val = train_df.loc[val_mask, features]
    y_val = train_df.loc[val_mask, TARGET_COLUMN]
    print(f"  Train: {X_train.shape[0]} rows, Val: {X_val.shape[0]} rows")

    # Train models on RAW target (no log transform)
    print("\n[3/4] Training models on raw target...")
    results = {}

    # --- LightGBM ---
    print(f"\n{'='*50}")
    print("Training: LightGBM V3 (raw target)")
    print(f"{'='*50}")

    lgb_model = lgb.LGBMRegressor(
        n_estimators=3000,
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
    t = time.time()
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(300)]
    )
    lgb_time = time.time() - t
    lgb_pred = np.clip(lgb_model.predict(X_val), 0, 1)
    lgb_r2 = r2_score(y_val, lgb_pred)
    lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_pred))
    results['LightGBM_V3'] = {'rmse': lgb_rmse, 'r2': lgb_r2, 'time': lgb_time}
    print(f"  LightGBM V3: RMSE={lgb_rmse:.6f}, R2={lgb_r2:.6f}, Score={max(0,100*lgb_r2):.2f}")

    # --- XGBoost ---
    print(f"\n{'='*50}")
    print("Training: XGBoost V3 (raw target)")
    print(f"{'='*50}")

    xgb_model = xgb.XGBRegressor(
        n_estimators=3000,
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
        verbosity=0,
        early_stopping_rounds=150
    )
    t = time.time()
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=300)
    xgb_time = time.time() - t
    xgb_pred = np.clip(xgb_model.predict(X_val), 0, 1)
    xgb_r2 = r2_score(y_val, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))
    results['XGBoost_V3'] = {'rmse': xgb_rmse, 'r2': xgb_r2, 'time': xgb_time}
    print(f"  XGBoost V3: RMSE={xgb_rmse:.6f}, R2={xgb_r2:.6f}, Score={max(0,100*xgb_r2):.2f}")

    # --- Optimized weighted ensemble ---
    print(f"\n{'='*50}")
    print("Optimizing weighted ensemble...")
    print(f"{'='*50}")

    best_w, best_r2 = 0.5, -1
    for w in np.arange(0.0, 1.01, 0.01):
        blend = w * lgb_pred + (1 - w) * xgb_pred
        r2 = r2_score(y_val, blend)
        if r2 > best_r2:
            best_r2 = r2
            best_w = w

    blend_pred = best_w * lgb_pred + (1 - best_w) * xgb_pred
    blend_rmse = np.sqrt(mean_squared_error(y_val, blend_pred))
    results['Ensemble_V3'] = {'rmse': blend_rmse, 'r2': best_r2, 'time': 0}
    print(f"  Best weight LGB={best_w:.2f}, XGB={1-best_w:.2f}")
    print(f"  Ensemble V3: RMSE={blend_rmse:.6f}, R2={best_r2:.6f}, Score={max(0,100*best_r2):.2f}")

    # Results summary
    print("\n" + "=" * 80)
    print("V3 FINAL RESULTS (score = R2 x 100)")
    print("=" * 80)
    print(f"{'Model':<25} {'RMSE':>12} {'R2':>10} {'Score':>8} {'Time':>8}")
    print("-" * 70)
    for name, m in sorted(results.items(), key=lambda x: -x[1]['r2']):
        score = max(0, 100 * m['r2'])
        print(f"{name:<25} {m['rmse']:>12.6f} {m['r2']:>10.6f} {score:>7.2f} {m['time']:>7.1f}s")
    print("-" * 70)

    # Pick best
    best_name = max(results, key=lambda k: results[k]['r2'])
    best_score = max(0, 100 * results[best_name]['r2'])
    print(f"\n>>> BEST: {best_name} (Score: {best_score:.2f}/100)")

    # Generate submission
    print("\n[4/4] Generating submission V3...")
    X_test = test_df[features]
    lgb_test = np.clip(lgb_model.predict(X_test), 0, 1)
    xgb_test = np.clip(xgb_model.predict(X_test), 0, 1)

    if 'Ensemble' in best_name:
        final_pred = best_w * lgb_test + (1 - best_w) * xgb_test
    elif 'LightGBM' in best_name:
        final_pred = lgb_test
    else:
        final_pred = xgb_test
    final_pred = np.clip(final_pred, 0, 1)

    submission = pd.DataFrame({'Index': test_df['Index'], 'demand': final_pred})
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v3.csv')
    submission.to_csv(out_path, index=False)

    # Verify
    sample = pd.read_csv(os.path.join(DATASET_DIR, 'sample_submission.csv'))
    assert list(submission.columns) == list(sample.columns)
    assert len(submission) == 41778
    print(f"  Saved: {out_path}")
    print(f"  Rows: {len(submission)}, Format: PASSED")
    print(f"  Predictions: min={final_pred.min():.6f}, max={final_pred.max():.6f}, mean={final_pred.mean():.6f}")

    # Save models
    joblib.dump(lgb_model, os.path.join(MODELS_DIR, 'lgb_model_v3.pkl'))
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_model_v3.pkl'))
    joblib.dump({
        'features': features, 'geo_stats': geo_stats,
        'best_weight': best_w, 'best_model': best_name,
        'use_log_transform': False,
    }, os.path.join(MODELS_DIR, 'model_meta_v3.pkl'))

    print("\n" + "=" * 60)
    print(f"V3 COMPLETE! Estimated Score: {best_score:.2f}/100")
    print(f"Upload '{out_path}' to HackerEarth")
    print("=" * 60)


if __name__ == '__main__':
    train_v3()
