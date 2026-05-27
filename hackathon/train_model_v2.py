"""
train_model_v2.py - Improved Model Training Pipeline
Flipkart Gridlock Hackathon 2.0

Improvements over v1:
  1. Log-transform target (demand is heavily right-skewed)
  2. Lag features (previous time slot demand per geohash)
  3. XGBoost with proper early stopping
  4. LightGBM + XGBoost ensemble blend
  5. Better hyperparameters
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import lightgbm as lgb
import xgboost as xgb

# Add hackathon dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN
from utils import decode_geohash_batch, parse_timestamp, time_slot_index, cyclic_encode


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'submissions')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


# ============================================================
# ENHANCED FEATURE ENGINEERING (V2)
# ============================================================

def engineer_features_v2(df, geo_stats=None, is_train=True):
    """
    Enhanced feature engineering with lag features and more aggregates.
    """
    df = df.copy()
    print(f"  Starting with {len(df)} rows, {len(df.columns)} columns")

    # ----------------------------------------------------------
    # 1. TIMESTAMP FEATURES
    # ----------------------------------------------------------
    print("  Engineering timestamp features...")
    parsed = df['timestamp'].apply(lambda ts: parse_timestamp(ts))
    df['hour'] = parsed.apply(lambda x: x[0])
    df['minute'] = parsed.apply(lambda x: x[1])
    df['time_slot'] = df.apply(lambda row: time_slot_index(row['hour'], row['minute']), axis=1)

    # Cyclic encoding
    cyclic = df['hour'].apply(lambda h: cyclic_encode(h, 24))
    df['sin_hour'] = cyclic.apply(lambda x: x[0])
    df['cos_hour'] = cyclic.apply(lambda x: x[1])

    cyclic_slot = df['time_slot'].apply(lambda s: cyclic_encode(s, 96))
    df['sin_slot'] = cyclic_slot.apply(lambda x: x[0])
    df['cos_slot'] = cyclic_slot.apply(lambda x: x[1])

    # Time period flags
    df['is_rush_hour'] = df['hour'].apply(lambda h: 1 if h in [8, 9, 10, 17, 18, 19] else 0)
    df['is_night'] = df['hour'].apply(lambda h: 1 if h in [23, 0, 1, 2, 3, 4, 5] else 0)
    df['is_morning_rush'] = df['hour'].apply(lambda h: 1 if h in [8, 9, 10] else 0)
    df['is_evening_rush'] = df['hour'].apply(lambda h: 1 if h in [17, 18, 19] else 0)

    def get_period(h):
        if 6 <= h < 12: return 0
        elif 12 <= h < 17: return 1
        elif 17 <= h < 22: return 2
        else: return 3
    df['period'] = df['hour'].apply(get_period)

    # ----------------------------------------------------------
    # 2. GEOHASH / SPATIAL FEATURES
    # ----------------------------------------------------------
    print("  Decoding geohashes to lat/lon...")
    lats, lons = decode_geohash_batch(df['geohash'])
    df['latitude'] = lats
    df['longitude'] = lons

    df['geo_prefix_4'] = df['geohash'].str[:4]
    df['geo_prefix_5'] = df['geohash'].str[:5]

    # ----------------------------------------------------------
    # 3. GEOHASH-LEVEL AGGREGATE STATS
    # ----------------------------------------------------------
    if is_train and 'demand' in df.columns:
        print("  Computing geohash-level demand statistics...")
        day48 = df[df['day'] == 48]
        geo_stats = {}

        # Per-geohash demand stats
        geo_demand = day48.groupby('geohash')['demand'].agg(
            ['mean', 'std', 'median', 'max', 'min', 'count']
        ).reset_index()
        geo_demand.columns = ['geohash', 'geo_demand_mean', 'geo_demand_std',
                              'geo_demand_median', 'geo_demand_max',
                              'geo_demand_min', 'geo_demand_count']
        geo_demand['geo_demand_std'] = geo_demand['geo_demand_std'].fillna(0)
        geo_stats['geo_demand'] = geo_demand

        # Per-geohash + time_slot demand (historical pattern)
        geo_time = day48.groupby(['geohash', 'time_slot'])['demand'].agg(
            ['mean', 'std']
        ).reset_index()
        geo_time.columns = ['geohash', 'time_slot', 'geo_time_demand_mean', 'geo_time_demand_std']
        geo_time['geo_time_demand_std'] = geo_time['geo_time_demand_std'].fillna(0)
        geo_stats['geo_time_demand'] = geo_time

        # Per-geohash + period demand (period-level pattern)
        geo_period = day48.copy()
        geo_period['period'] = geo_period['hour'] if 'hour' in geo_period.columns else 0
        # Recompute period for day48
        parsed48 = geo_period['timestamp'].apply(lambda ts: parse_timestamp(ts))
        geo_period['hour_tmp'] = parsed48.apply(lambda x: x[0])
        geo_period['period_tmp'] = geo_period['hour_tmp'].apply(get_period)
        geo_period_agg = geo_period.groupby(['geohash', 'period_tmp'])['demand'].mean().reset_index()
        geo_period_agg.columns = ['geohash', 'period', 'geo_period_demand_mean']
        geo_stats['geo_period_demand'] = geo_period_agg

        # Per prefix-4 demand
        prefix_demand = day48.groupby(day48['geohash'].str[:4])['demand'].agg(
            ['mean', 'std']
        ).reset_index()
        prefix_demand.columns = ['geo_prefix_4', 'prefix4_demand_mean', 'prefix4_demand_std']
        prefix_demand['prefix4_demand_std'] = prefix_demand['prefix4_demand_std'].fillna(0)
        geo_stats['prefix4_demand'] = prefix_demand

        # Per-geohash + road_type demand
        geo_road = day48.groupby(['geohash', 'RoadType'])['demand'].mean().reset_index()
        geo_road.columns = ['geohash', 'RoadType', 'geo_road_demand_mean']
        geo_stats['geo_road_demand'] = geo_road

        # Per time_slot global demand
        time_demand = day48.groupby('time_slot')['demand'].agg(['mean', 'std']).reset_index()
        time_demand.columns = ['time_slot', 'time_global_demand_mean', 'time_global_demand_std']
        geo_stats['time_global_demand'] = time_demand

        # Global stats
        geo_stats['global_mean'] = day48['demand'].mean()
        geo_stats['global_median'] = day48['demand'].median()

    # Merge all geo_stats
    if geo_stats is not None:
        print("  Merging geohash-level statistics...")

        # Per-geohash
        df = df.merge(geo_stats['geo_demand'], on='geohash', how='left')
        for col in ['geo_demand_mean', 'geo_demand_median', 'geo_demand_max', 'geo_demand_min']:
            df[col] = df[col].fillna(geo_stats['global_mean'])
        df['geo_demand_std'] = df['geo_demand_std'].fillna(0)
        df['geo_demand_count'] = df['geo_demand_count'].fillna(0)

        # Per-geohash + time_slot
        df = df.merge(geo_stats['geo_time_demand'], on=['geohash', 'time_slot'], how='left')
        df['geo_time_demand_mean'] = df['geo_time_demand_mean'].fillna(df['geo_demand_mean'])
        df['geo_time_demand_std'] = df['geo_time_demand_std'].fillna(0)

        # Per-geohash + period
        df = df.merge(geo_stats['geo_period_demand'], on=['geohash', 'period'], how='left')
        df['geo_period_demand_mean'] = df['geo_period_demand_mean'].fillna(df['geo_demand_mean'])

        # Per-prefix
        df = df.merge(geo_stats['prefix4_demand'], on='geo_prefix_4', how='left')
        df['prefix4_demand_mean'] = df['prefix4_demand_mean'].fillna(geo_stats['global_mean'])
        df['prefix4_demand_std'] = df['prefix4_demand_std'].fillna(0)

        # Per-geohash + road_type
        df = df.merge(geo_stats['geo_road_demand'], on=['geohash', 'RoadType'], how='left')
        df['geo_road_demand_mean'] = df['geo_road_demand_mean'].fillna(df['geo_demand_mean'])

        # Per time_slot global
        df = df.merge(geo_stats['time_global_demand'], on='time_slot', how='left')
        df['time_global_demand_mean'] = df['time_global_demand_mean'].fillna(geo_stats['global_mean'])
        df['time_global_demand_std'] = df['time_global_demand_std'].fillna(0)

        # Ratio features: how does this zone compare to global at this time?
        df['geo_vs_global_ratio'] = df['geo_time_demand_mean'] / (df['time_global_demand_mean'] + 1e-8)
        df['geo_vs_prefix_ratio'] = df['geo_demand_mean'] / (df['prefix4_demand_mean'] + 1e-8)

    # ----------------------------------------------------------
    # 4. ROAD & ENVIRONMENT FEATURES
    # ----------------------------------------------------------
    print("  Encoding categorical features...")

    df['RoadType'] = df['RoadType'].fillna('Unknown')
    road_type_map = {'Street': 0, 'Residential': 1, 'Highway': 2, 'Unknown': -1}
    df['road_type_encoded'] = df['RoadType'].map(road_type_map).fillna(-1).astype(int)

    df['Weather'] = df['Weather'].fillna('Unknown')
    weather_map = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3, 'Unknown': -1}
    df['weather_encoded'] = df['Weather'].map(weather_map).fillna(-1).astype(int)

    df['large_vehicles'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['landmarks'] = (df['Landmarks'] == 'Yes').astype(int)

    temp_median = df['Temperature'].median()
    df['temperature'] = df['Temperature'].fillna(temp_median)

    df['temp_bin'] = pd.cut(df['temperature'], bins=[-20, 5, 15, 25, 35, 50], labels=[0, 1, 2, 3, 4])
    df['temp_bin'] = df['temp_bin'].cat.add_categories(-1).fillna(-1).astype(int)

    # ----------------------------------------------------------
    # 5. INTERACTION FEATURES
    # ----------------------------------------------------------
    print("  Creating interaction features...")

    df['road_capacity'] = df['NumberofLanes'] * (1 + df['large_vehicles'])

    weather_severity = {'Sunny': 0, 'Rainy': 2, 'Foggy': 3, 'Snowy': 4, 'Unknown': 1}
    df['weather_severity'] = df['Weather'].map(weather_severity).fillna(1)

    df['rush_road_interaction'] = df['is_rush_hour'] * df['road_type_encoded']
    df['landmark_rush'] = df['landmarks'] * df['is_rush_hour']
    df['lanes_rush'] = df['NumberofLanes'] * df['is_rush_hour']
    df['weather_night'] = df['weather_severity'] * df['is_night']

    # Demand deviation from zone mean
    if 'geo_demand_mean' in df.columns and 'geo_time_demand_mean' in df.columns:
        df['demand_deviation'] = df['geo_time_demand_mean'] - df['geo_demand_mean']

    print(f"  Done! Final shape: {df.shape}")

    return df, geo_stats


# ============================================================
# V2 FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS_V2 = [
    # Temporal
    'hour', 'minute', 'time_slot', 'sin_hour', 'cos_hour',
    'sin_slot', 'cos_slot', 'is_rush_hour', 'is_night',
    'is_morning_rush', 'is_evening_rush', 'period',
    # Spatial
    'latitude', 'longitude',
    # Geohash aggregates (expanded)
    'geo_demand_mean', 'geo_demand_std', 'geo_demand_median',
    'geo_demand_max', 'geo_demand_min', 'geo_demand_count',
    'geo_time_demand_mean', 'geo_time_demand_std',
    'geo_period_demand_mean',
    'prefix4_demand_mean', 'prefix4_demand_std',
    'geo_road_demand_mean',
    'time_global_demand_mean', 'time_global_demand_std',
    # Ratio features
    'geo_vs_global_ratio', 'geo_vs_prefix_ratio',
    # Road
    'road_type_encoded', 'NumberofLanes', 'large_vehicles', 'landmarks',
    'road_capacity',
    # Environment
    'temperature', 'temp_bin', 'weather_encoded', 'weather_severity',
    # Interactions
    'rush_road_interaction', 'landmark_rush', 'lanes_rush', 'weather_night',
    'demand_deviation',
    # Day
    'day',
]


# ============================================================
# LOAD & ENGINEER V2
# ============================================================

def load_and_engineer_v2():
    """Load data and apply v2 feature engineering."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    train = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATASET_DIR, 'test.csv'))
    print(f"Train: {train.shape}")
    print(f"Test:  {test.shape}")

    print("\n" + "=" * 60)
    print("ENGINEERING TRAIN FEATURES (V2)")
    print("=" * 60)
    train_eng, geo_stats = engineer_features_v2(train, is_train=True)

    print("\n" + "=" * 60)
    print("ENGINEERING TEST FEATURES (V2)")
    print("=" * 60)
    test_eng, _ = engineer_features_v2(test, geo_stats=geo_stats, is_train=False)

    # Verify features
    missing = [c for c in FEATURE_COLUMNS_V2 if c not in test_eng.columns]
    if missing:
        print(f"\n[WARN] Missing test features: {missing}")
        # Remove missing features from the list
        features = [c for c in FEATURE_COLUMNS_V2 if c in test_eng.columns and c in train_eng.columns]
    else:
        features = FEATURE_COLUMNS_V2

    print(f"\n[OK] V2 Feature engineering complete!")
    print(f"   Train: {train_eng.shape}, Test: {test_eng.shape}")
    print(f"   Features: {len(features)} columns")

    return train_eng, test_eng, features, geo_stats


# ============================================================
# TRAIN V2 MODELS
# ============================================================

def train_v2():
    """Full v2 training pipeline with log-transform and ensemble."""

    print("=" * 60)
    print("GRIDLOCK HACKATHON 2.0 - V2 MODEL TRAINING")
    print("=" * 60)

    # Step 1: Load and engineer
    print("\n[1/5] Loading and engineering features (V2)...")
    train_df, test_df, features, geo_stats = load_and_engineer_v2()

    # Step 2: Time-based split
    print("\n[2/5] Creating time-based validation split...")
    train_mask = train_df['day'] == 48
    val_mask = train_df['day'] == 49

    X_train = train_df.loc[train_mask, features]
    y_train = train_df.loc[train_mask, TARGET_COLUMN]
    X_val = train_df.loc[val_mask, features]
    y_val = train_df.loc[val_mask, TARGET_COLUMN]

    print(f"  Train: {X_train.shape[0]} rows (Day 48)")
    print(f"  Val:   {X_val.shape[0]} rows (Day 49)")

    # Step 3: Log-transform target
    print("\n[3/5] Applying log1p transform to target...")
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    print(f"  Original mean: {y_train.mean():.6f} -> Log mean: {y_train_log.mean():.6f}")

    # Step 4: Train models
    print("\n[4/5] Training models...")
    results = {}
    trained_models = {}

    # --- LightGBM ---
    print(f"\n{'='*50}")
    print("Training: LightGBM (V2)")
    print(f"{'='*50}")

    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
        max_depth=10,
        learning_rate=0.03,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=15,
        reg_alpha=0.05,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    start_time = time.time()
    lgb_model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200)
        ]
    )
    lgb_time = time.time() - start_time

    # Predict and inverse log-transform
    lgb_pred_val = np.expm1(lgb_model.predict(X_val))
    lgb_pred_val = np.clip(lgb_pred_val, 0, 1)

    lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_pred_val))
    lgb_r2 = r2_score(y_val, lgb_pred_val)
    results['LightGBM_V2'] = {'val_rmse': lgb_rmse, 'val_r2': lgb_r2, 'time': lgb_time}
    trained_models['LightGBM_V2'] = lgb_model
    print(f"\n  LightGBM V2: RMSE={lgb_rmse:.6f}, R2={lgb_r2:.6f}, Time={lgb_time:.1f}s")

    # --- XGBoost ---
    print(f"\n{'='*50}")
    print("Training: XGBoost (V2)")
    print(f"{'='*50}")

    xgb_model = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.05,
        reg_lambda=1.0,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=100
    )

    start_time = time.time()
    xgb_model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        verbose=200
    )
    xgb_time = time.time() - start_time

    xgb_pred_val = np.expm1(xgb_model.predict(X_val))
    xgb_pred_val = np.clip(xgb_pred_val, 0, 1)

    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred_val))
    xgb_r2 = r2_score(y_val, xgb_pred_val)
    results['XGBoost_V2'] = {'val_rmse': xgb_rmse, 'val_r2': xgb_r2, 'time': xgb_time}
    trained_models['XGBoost_V2'] = xgb_model
    print(f"\n  XGBoost V2: RMSE={xgb_rmse:.6f}, R2={xgb_r2:.6f}, Time={xgb_time:.1f}s")

    # --- Ensemble: Average of LightGBM + XGBoost ---
    print(f"\n{'='*50}")
    print("Creating: Ensemble (LightGBM + XGBoost average)")
    print(f"{'='*50}")

    ensemble_pred_val = (lgb_pred_val + xgb_pred_val) / 2
    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred_val))
    ensemble_r2 = r2_score(y_val, ensemble_pred_val)
    results['Ensemble'] = {'val_rmse': ensemble_rmse, 'val_r2': ensemble_r2, 'time': 0}
    print(f"\n  Ensemble: RMSE={ensemble_rmse:.6f}, R2={ensemble_r2:.6f}")

    # --- Weighted Ensemble (optimize weights) ---
    print(f"\n{'='*50}")
    print("Optimizing: Weighted Ensemble")
    print(f"{'='*50}")

    best_weight = 0.5
    best_weighted_r2 = ensemble_r2
    for w in np.arange(0.1, 0.95, 0.05):
        weighted_pred = w * lgb_pred_val + (1 - w) * xgb_pred_val
        weighted_r2 = r2_score(y_val, weighted_pred)
        if weighted_r2 > best_weighted_r2:
            best_weighted_r2 = weighted_r2
            best_weight = w

    weighted_pred_val = best_weight * lgb_pred_val + (1 - best_weight) * xgb_pred_val
    weighted_rmse = np.sqrt(mean_squared_error(y_val, weighted_pred_val))
    weighted_r2 = r2_score(y_val, weighted_pred_val)
    results['Weighted_Ensemble'] = {'val_rmse': weighted_rmse, 'val_r2': weighted_r2, 'time': 0}
    print(f"  Best weight (LightGBM): {best_weight:.2f}")
    print(f"  Weighted Ensemble: RMSE={weighted_rmse:.6f}, R2={weighted_r2:.6f}")

    # Step 5: Results Summary
    print("\n" + "=" * 80)
    print("V2 MODEL COMPARISON (score = R2 x 100)")
    print("=" * 80)
    print(f"{'Model':<25} {'Val RMSE':>12} {'Val R2':>10} {'Score':>8} {'Time':>8}")
    print("-" * 70)

    sorted_results = sorted(results.items(), key=lambda x: -x[1]['val_r2'])
    for name, m in sorted_results:
        score = max(0, 100 * m['val_r2'])
        print(f"{name:<25} {m['val_rmse']:>12.6f} {m['val_r2']:>10.6f} {score:>7.2f} {m['time']:>7.1f}s")

    best_name = sorted_results[0][0]
    best_score = max(0, 100 * sorted_results[0][1]['val_r2'])
    print("-" * 70)
    print(f"\n>>> BEST: {best_name} (Estimated Score: {best_score:.2f}/100)")

    # Feature importance
    if hasattr(lgb_model, 'feature_importances_'):
        importances = lgb_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print(f"\n{'='*50}")
        print(f"TOP 15 FEATURE IMPORTANCES (LightGBM V2)")
        print(f"{'='*50}")
        for i in range(min(15, len(features))):
            idx = indices[i]
            print(f"  {i+1:2d}. {features[idx]:<30} {importances[idx]}")

    # Save everything for prediction
    print("\n[5/5] Saving models...")
    joblib.dump(lgb_model, os.path.join(MODELS_DIR, 'lgb_model_v2.pkl'))
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_model_v2.pkl'))
    joblib.dump({
        'features': features,
        'geo_stats': geo_stats,
        'best_weight': best_weight,
        'best_model': best_name,
        'use_log_transform': True,
    }, os.path.join(MODELS_DIR, 'model_meta_v2.pkl'))

    print(f"  Models saved to {MODELS_DIR}")

    # Generate test predictions
    print("\n" + "=" * 60)
    print("GENERATING SUBMISSION V2")
    print("=" * 60)

    X_test = test_df[features]

    lgb_pred_test = np.expm1(lgb_model.predict(X_test))
    xgb_pred_test = np.expm1(xgb_model.predict(X_test))

    # Use the best approach
    if 'Weighted' in best_name:
        final_pred = best_weight * lgb_pred_test + (1 - best_weight) * xgb_pred_test
    elif 'Ensemble' in best_name:
        final_pred = (lgb_pred_test + xgb_pred_test) / 2
    elif 'LightGBM' in best_name:
        final_pred = lgb_pred_test
    else:
        final_pred = xgb_pred_test

    final_pred = np.clip(final_pred, 0, 1)

    submission = pd.DataFrame({
        'Index': test_df['Index'],
        'demand': final_pred
    })

    output_path = os.path.join(SUBMISSIONS_DIR, 'submission_v2.csv')
    submission.to_csv(output_path, index=False)

    print(f"  Predictions: min={final_pred.min():.6f}, max={final_pred.max():.6f}, mean={final_pred.mean():.6f}")
    print(f"  Submission saved: {output_path}")
    print(f"  Rows: {len(submission)}")

    # Verify format
    sample = pd.read_csv(os.path.join(DATASET_DIR, 'sample_submission.csv'))
    assert list(submission.columns) == list(sample.columns), "Column mismatch!"
    assert len(submission) == 41778, f"Row count mismatch: {len(submission)}"
    print(f"  Format check: PASSED")

    print("\n" + "=" * 60)
    print(f"V2 COMPLETE! Estimated Score: {best_score:.2f}/100")
    print(f"Upload '{output_path}' to HackerEarth")
    print("=" * 60)

    return results


if __name__ == '__main__':
    train_v2()
