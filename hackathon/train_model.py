"""
train_model.py - Train & Evaluate ML Models
Flipkart Gridlock Hackathon 2.0

Trains multiple models on engineered features, evaluates using
time-based validation, and saves the best model.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional: LightGBM and XGBoost
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: LightGBM not installed. Skipping LightGBM model.")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: XGBoost not installed. Skipping XGBoost model.")

# Add hackathon dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import load_and_engineer, FEATURE_COLUMNS, TARGET_COLUMN


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# VALIDATION SPLIT (Time-Based)
# ============================================================

def time_based_split(df, features, target):
    """
    Split data by time: Day 48 -> train, Day 49 -> validation.
    This mimics the actual test scenario (predict future from past).
    
    Args:
        df: Engineered DataFrame with 'day' column
        features: List of feature column names
        target: Target column name
    
    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    train_mask = df['day'] == 48
    val_mask = df['day'] == 49
    
    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, target]
    X_val = df.loc[val_mask, features]
    y_val = df.loc[val_mask, target]
    
    print(f"  Train: {X_train.shape[0]} rows (Day 48)")
    print(f"  Val:   {X_val.shape[0]} rows (Day 49)")
    
    return X_train, X_val, y_train, y_val


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_models():
    """
    Returns a dictionary of models to train and evaluate.
    
    Returns:
        dict: {model_name: model_instance}
    """
    models = {}
    
    # 1. Linear Regression (baseline)
    models['Linear Regression'] = LinearRegression()
    
    # 2. Random Forest
    models['Random Forest'] = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    # 3. Gradient Boosting (your existing approach!)
    models['Gradient Boosting'] = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    # 4. LightGBM (fast, handles categoricals)
    if HAS_LGBM:
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    # 5. XGBoost (industry standard)
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    
    return models


# ============================================================
# TRAIN & EVALUATE
# ============================================================

def train_and_evaluate(models, X_train, X_val, y_train, y_val):
    """
    Train all models and evaluate on validation set.
    
    Args:
        models: Dict of {name: model_instance}
        X_train, X_val: Feature DataFrames
        y_train, y_val: Target Series
    
    Returns:
        tuple: (results_dict, trained_models_dict)
    """
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # Train
        if name in ['LightGBM'] and HAS_LGBM:
            # LightGBM with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=100)
                ]
            )
        elif name in ['XGBoost'] and HAS_XGB:
            # XGBoost with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=100
            )
        else:
            model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        # Clip predictions to valid range
        y_pred_train = np.clip(y_pred_train, 0, 1)
        y_pred_val = np.clip(y_pred_val, 0, 1)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_r2 = r2_score(y_val, y_pred_val)
        
        results[name] = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'train_time': train_time
        }
        trained_models[name] = model
        
        print(f"\n  Results for {name}:")
        print(f"    Train RMSE:  {train_rmse:.6f}")
        print(f"    Val RMSE:    {val_rmse:.6f}")
        print(f"    Val MAE:     {val_mae:.6f}")
        print(f"    Val R2:      {val_r2:.6f}")
        print(f"    Train Time:  {train_time:.1f}s")
    
    return results, trained_models


# ============================================================
# RESULTS SUMMARY
# ============================================================

def print_results_summary(results):
    """Print a formatted comparison table of all models."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Model':<22} {'Train RMSE':>12} {'Val RMSE':>12} {'Val MAE':>12} {'Val R2':>10} {'Time':>8}")
    print("-" * 80)
    
    # Sort by val_rmse (lower is better)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['val_rmse'])
    
    for name, metrics in sorted_results:
        print(f"{name:<22} {metrics['train_rmse']:>12.6f} {metrics['val_rmse']:>12.6f} "
              f"{metrics['val_mae']:>12.6f} {metrics['val_r2']:>10.6f} {metrics['train_time']:>7.1f}s")
    
    best_model = sorted_results[0][0]
    best_rmse = sorted_results[0][1]['val_rmse']
    print("-" * 80)
    print(f"\n>>> BEST MODEL: {best_model} (Val RMSE: {best_rmse:.6f})")
    
    return best_model


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_model(model, name, geo_stats, features):
    """Save the trained model and metadata for prediction."""
    save_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    meta_path = os.path.join(MODELS_DIR, 'model_meta.pkl')
    
    joblib.dump(model, save_path)
    joblib.dump({
        'model_name': name,
        'geo_stats': geo_stats,
        'features': features
    }, meta_path)
    
    print(f"\nModel saved to: {save_path}")
    print(f"Metadata saved to: {meta_path}")


# ============================================================
# FEATURE IMPORTANCE (for tree-based models)
# ============================================================

def print_feature_importance(model, features, model_name, top_n=15):
    """Print top N most important features."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\n{'='*50}")
        print(f"TOP {top_n} FEATURE IMPORTANCES ({model_name})")
        print(f"{'='*50}")
        
        for i in range(min(top_n, len(features))):
            idx = indices[i]
            print(f"  {i+1:2d}. {features[idx]:<25} {importances[idx]:.4f}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("GRIDLOCK HACKATHON 2.0 - MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Load and engineer features
    print("\n[1/4] Loading and engineering features...")
    train_df, test_df, features, geo_stats = load_and_engineer()
    
    # Step 2: Time-based split
    print("\n[2/4] Creating time-based validation split...")
    X_train, X_val, y_train, y_val = time_based_split(
        train_df, features, TARGET_COLUMN
    )
    
    # Step 3: Train models
    print("\n[3/4] Training models...")
    models = get_models()
    results, trained_models = train_and_evaluate(
        models, X_train, X_val, y_train, y_val
    )
    
    # Step 4: Results & save best
    best_model_name = print_results_summary(results)
    best_model = trained_models[best_model_name]
    
    # Feature importance
    print_feature_importance(best_model, features, best_model_name)
    
    # Save
    print("\n[4/4] Saving best model...")
    save_model(best_model, best_model_name, geo_stats, features)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Best model: {best_model_name}")
    print(f"Next step: Run predict.py to generate submission")
    print("=" * 60)
    
    return best_model, results, geo_stats


if __name__ == '__main__':
    main()
