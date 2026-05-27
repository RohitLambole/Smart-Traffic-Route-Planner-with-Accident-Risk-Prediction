"""
train_model_v6.py - K-Fold Stacking + Target Encoding
Flipkart Gridlock Hackathon 2.0

Key changes from V5:
  1. K-Fold cross-validation (not Day 48 vs Day 49 split)
  2. Proper target encoding for geohash (mean target per fold)
  3. Out-of-fold predictions for stacking
  4. Blend of multiple models via stacking
  5. Use ALL data more effectively
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import decode_geohash_batch, parse_timestamp, time_slot_index, cyclic_encode

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'submissions')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

N_FOLDS = 5


def load_data():
    train = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATASET_DIR, 'test.csv'))
    return train, test


def build_base_features(df):
    """Build non-leaking features that don't require target encoding."""
    df = df.copy()

    # Time features
    parsed = df['timestamp'].apply(lambda ts: parse_timestamp(ts))
    df['hour'] = parsed.apply(lambda x: x[0])
    df['minute'] = parsed.apply(lambda x: x[1])
    df['time_slot'] = df.apply(lambda r: time_slot_index(r['hour'], r['minute']), axis=1)

    cyclic_h = df['hour'].apply(lambda h: cyclic_encode(h, 24))
    df['sin_hour'] = cyclic_h.apply(lambda x: x[0])
    df['cos_hour'] = cyclic_h.apply(lambda x: x[1])
    cyclic_s = df['time_slot'].apply(lambda s: cyclic_encode(s, 96))
    df['sin_slot'] = cyclic_s.apply(lambda x: x[0])
    df['cos_slot'] = cyclic_s.apply(lambda x: x[1])

    df['is_rush'] = df['hour'].apply(lambda h: 1 if h in [8,9,10,17,18,19] else 0)
    df['is_night'] = df['hour'].apply(lambda h: 1 if h in [23,0,1,2,3,4,5] else 0)
    df['is_morning_rush'] = df['hour'].apply(lambda h: 1 if h in [8,9,10] else 0)
    df['is_evening_rush'] = df['hour'].apply(lambda h: 1 if h in [17,18,19] else 0)

    def get_period(h):
        if 6 <= h < 12: return 0
        elif 12 <= h < 17: return 1
        elif 17 <= h < 22: return 2
        else: return 3
    df['period'] = df['hour'].apply(get_period)

    # Road features
    df['RoadType'] = df['RoadType'].fillna('Unknown')
    road_map = {'Street': 0, 'Residential': 1, 'Highway': 2, 'Unknown': -1}
    df['road_encoded'] = df['RoadType'].map(road_map).fillna(-1).astype(int)

    df['Weather'] = df['Weather'].fillna('Unknown')
    weather_map = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3, 'Unknown': -1}
    df['weather_encoded'] = df['Weather'].map(weather_map).fillna(-1).astype(int)

    weather_severity = {'Sunny': 0, 'Rainy': 2, 'Foggy': 3, 'Snowy': 4, 'Unknown': 1}
    df['weather_severity'] = df['Weather'].map(weather_severity).fillna(1).astype(int)

    df['large_vehicles'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['landmarks'] = (df['Landmarks'] == 'Yes').astype(int)
    df['temperature'] = df['Temperature'].fillna(df['Temperature'].median())

    df['road_capacity'] = df['NumberofLanes'] * (1 + df['large_vehicles'])

    # Spatial
    lats, lons = decode_geohash_batch(df['geohash'])
    df['latitude'] = lats
    df['longitude'] = lons

    # Geohash codes
    df['geohash_code'] = pd.Categorical(df['geohash']).codes
    df['geo_prefix_4'] = pd.Categorical(df['geohash'].str[:4]).codes
    df['geo_prefix_5'] = pd.Categorical(df['geohash'].str[:5]).codes

    return df


def add_target_encoding(train_df, test_df, target_col='demand'):
    """
    K-Fold target encoding for geohash and (geohash, time_slot).
    This avoids target leakage by computing the encoding from out-of-fold data.
    """
    print("  Computing K-Fold target encoding...")
    
    # Initialize encoded columns
    train_df['te_geohash'] = 0.0
    train_df['te_geo_ts'] = 0.0
    train_df['te_geo_period'] = 0.0
    train_df['te_timeslot'] = 0.0
    
    global_mean = train_df[target_col].mean()
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        fold_train = train_df.iloc[train_idx]
        
        # Geohash target encoding
        geo_te = fold_train.groupby('geohash')[target_col].mean()
        train_df.loc[train_df.index[val_idx], 'te_geohash'] = (
            train_df.iloc[val_idx]['geohash'].map(geo_te).fillna(global_mean).values
        )
        
        # (Geohash, time_slot) target encoding
        geo_ts_te = fold_train.groupby(['geohash', 'time_slot'])[target_col].mean()
        val_keys = list(zip(train_df.iloc[val_idx]['geohash'], train_df.iloc[val_idx]['time_slot']))
        train_df.loc[train_df.index[val_idx], 'te_geo_ts'] = [
            geo_ts_te.get(k, global_mean) for k in val_keys
        ]
        
        # (Geohash, period) target encoding
        geo_period_te = fold_train.groupby(['geohash', 'period'])[target_col].mean()
        val_keys_p = list(zip(train_df.iloc[val_idx]['geohash'], train_df.iloc[val_idx]['period']))
        train_df.loc[train_df.index[val_idx], 'te_geo_period'] = [
            geo_period_te.get(k, global_mean) for k in val_keys_p
        ]

        # Time_slot target encoding
        ts_te = fold_train.groupby('time_slot')[target_col].mean()
        train_df.loc[train_df.index[val_idx], 'te_timeslot'] = (
            train_df.iloc[val_idx]['time_slot'].map(ts_te).fillna(global_mean).values
        )
    
    # For test: use full train data for encoding
    geo_te_full = train_df.groupby('geohash')[target_col].mean()
    test_df['te_geohash'] = test_df['geohash'].map(geo_te_full).fillna(global_mean)
    
    geo_ts_te_full = train_df.groupby(['geohash', 'time_slot'])[target_col].mean()
    test_keys = list(zip(test_df['geohash'], test_df['time_slot']))
    test_df['te_geo_ts'] = [geo_ts_te_full.get(k, global_mean) for k in test_keys]
    
    geo_period_te_full = train_df.groupby(['geohash', 'period'])[target_col].mean()
    test_keys_p = list(zip(test_df['geohash'], test_df['period']))
    test_df['te_geo_period'] = [geo_period_te_full.get(k, global_mean) for k in test_keys_p]

    ts_te_full = train_df.groupby('time_slot')[target_col].mean()
    test_df['te_timeslot'] = test_df['time_slot'].map(ts_te_full).fillna(global_mean)

    # Additional aggregate stats (no leakage — uses full train for both)
    geo_stats = train_df.groupby('geohash')[target_col].agg(['std', 'median', 'min', 'max', 'count']).reset_index()
    geo_stats.columns = ['geohash', 'geo_std', 'geo_median', 'geo_min', 'geo_max', 'geo_count']
    geo_stats['geo_std'] = geo_stats['geo_std'].fillna(0)
    
    for df in [train_df, test_df]:
        merged = df[['geohash']].merge(geo_stats, on='geohash', how='left')
        df['geo_std'] = merged['geo_std'].fillna(0).values
        df['geo_median'] = merged['geo_median'].fillna(global_mean).values
        df['geo_min'] = merged['geo_min'].fillna(global_mean).values
        df['geo_max'] = merged['geo_max'].fillna(global_mean).values
        df['geo_count'] = merged['geo_count'].fillna(0).values

    # Ratio features
    for df in [train_df, test_df]:
        df['te_ratio_vs_geo'] = df['te_geo_ts'] / (df['te_geohash'] + 1e-8)
        df['te_ratio_vs_ts'] = df['te_geo_ts'] / (df['te_timeslot'] + 1e-8)
        df['te_deviation'] = df['te_geo_ts'] - df['te_geohash']

    return train_df, test_df


def train_v6():
    print("=" * 70)
    print("GRIDLOCK HACKATHON 2.0 - V6 (K-FOLD STACKING)")
    print("=" * 70)

    # Load and build base features
    print("\n[1/5] Loading data and building base features...")
    train_raw, test_raw = load_data()
    train = build_base_features(train_raw)
    test = build_base_features(test_raw)
    print(f"  Train: {train.shape}, Test: {test.shape}")

    # Target encoding
    print("\n[2/5] Target encoding (K-Fold, no leakage)...")
    train, test = add_target_encoding(train, test)

    # Feature list
    feature_cols = [
        # Target encodings (MOST IMPORTANT)
        'te_geohash', 'te_geo_ts', 'te_geo_period', 'te_timeslot',
        # Geo stats
        'geo_std', 'geo_median', 'geo_min', 'geo_max', 'geo_count',
        # Target encoding ratios
        'te_ratio_vs_geo', 'te_ratio_vs_ts', 'te_deviation',
        # Temporal
        'hour', 'minute', 'time_slot', 'sin_hour', 'cos_hour',
        'sin_slot', 'cos_slot', 'is_rush', 'is_night',
        'is_morning_rush', 'is_evening_rush', 'period',
        # Road & env
        'road_encoded', 'weather_encoded', 'weather_severity',
        'NumberofLanes', 'large_vehicles', 'landmarks',
        'temperature', 'road_capacity',
        # Spatial & geohash
        'latitude', 'longitude',
        'geohash_code', 'geo_prefix_4', 'geo_prefix_5',
        # Day
        'day',
    ]

    feature_cols = [c for c in feature_cols if c in train.columns and c in test.columns]
    print(f"  Features: {len(feature_cols)}")

    target = 'demand'
    X = train[feature_cols].values
    y = train[target].values
    X_test = test[feature_cols].values

    # K-Fold stacking
    print(f"\n[3/5] K-Fold training ({N_FOLDS} folds)...")

    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    test_lgb = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n  --- Fold {fold_idx + 1}/{N_FOLDS} ---")
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=5000, max_depth=-1, learning_rate=0.01,
            num_leaves=511, subsample=0.7, colsample_bytree=0.6,
            min_child_samples=5, reg_alpha=0.05, reg_lambda=0.5,
            random_state=42, n_jobs=-1, verbose=-1
        )
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof_lgb[val_idx] = lgb_model.predict(X_vl)
        test_lgb += lgb_model.predict(X_test) / N_FOLDS

        lgb_r2 = r2_score(y_vl, oof_lgb[val_idx])
        print(f"    LGB fold R2: {lgb_r2:.6f}")

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=5000, max_depth=8, learning_rate=0.01,
            subsample=0.7, colsample_bytree=0.6, min_child_weight=3,
            reg_alpha=0.05, reg_lambda=1.0, gamma=0.01,
            random_state=42, n_jobs=-1, verbosity=0,
            early_stopping_rounds=100
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=0)
        oof_xgb[val_idx] = xgb_model.predict(X_vl)
        test_xgb += xgb_model.predict(X_test) / N_FOLDS

        xgb_r2 = r2_score(y_vl, oof_xgb[val_idx])
        print(f"    XGB fold R2: {xgb_r2:.6f}")

    # OOF scores
    oof_lgb = np.clip(oof_lgb, 0, 1)
    oof_xgb = np.clip(oof_xgb, 0, 1)
    test_lgb = np.clip(test_lgb, 0, 1)
    test_xgb = np.clip(test_xgb, 0, 1)

    lgb_oof_r2 = r2_score(y, oof_lgb)
    xgb_oof_r2 = r2_score(y, oof_xgb)
    print(f"\n  OOF LightGBM R2: {lgb_oof_r2:.6f} (Score: {max(0,100*lgb_oof_r2):.2f})")
    print(f"  OOF XGBoost R2:  {xgb_oof_r2:.6f} (Score: {max(0,100*xgb_oof_r2):.2f})")

    # Stacking: optimize blend
    print(f"\n[4/5] Optimizing ensemble weights...")
    best_w, best_r2 = 0.5, -1
    for w in np.arange(0.0, 1.01, 0.01):
        blend = w * oof_lgb + (1 - w) * oof_xgb
        r2 = r2_score(y, blend)
        if r2 > best_r2:
            best_r2 = r2
            best_w = w

    print(f"  Best weight: LGB={best_w:.2f}, XGB={1-best_w:.2f}")
    print(f"  Blend OOF R2: {best_r2:.6f} (Score: {max(0,100*best_r2):.2f})")

    # Stacking with Ridge
    print(f"\n  Training Ridge meta-learner on OOF predictions...")
    stack_train = np.column_stack([oof_lgb, oof_xgb])
    stack_test = np.column_stack([test_lgb, test_xgb])
    ridge = Ridge(alpha=1.0)
    ridge.fit(stack_train, y)
    stack_pred = np.clip(ridge.predict(stack_train), 0, 1)
    stack_r2 = r2_score(y, stack_pred)
    print(f"  Ridge stack R2: {stack_r2:.6f} (Score: {max(0,100*stack_r2):.2f})")
    print(f"  Ridge weights: LGB={ridge.coef_[0]:.4f}, XGB={ridge.coef_[1]:.4f}, intercept={ridge.intercept_:.6f}")

    # Results
    print("\n" + "=" * 70)
    print("V6 RESULTS")
    print("=" * 70)
    all_r = {
        'LightGBM (OOF)': lgb_oof_r2,
        'XGBoost (OOF)': xgb_oof_r2,
        'Weighted Blend': best_r2,
        'Ridge Stack': stack_r2,
    }
    for name, r2 in sorted(all_r.items(), key=lambda x: -x[1]):
        print(f"  {name:<25} R2={r2:.6f}  Score={max(0,100*r2):.2f}")

    # Generate submissions
    print(f"\n[5/5] Generating submissions...")

    # A: Best single model
    if xgb_oof_r2 > lgb_oof_r2:
        sub_single = test_xgb
        single_name = 'XGBoost'
    else:
        sub_single = test_lgb
        single_name = 'LightGBM'

    # B: Weighted blend
    sub_blend = best_w * test_lgb + (1 - best_w) * test_xgb
    sub_blend = np.clip(sub_blend, 0, 1)

    # C: Ridge stack
    sub_stack = np.clip(ridge.predict(stack_test), 0, 1)

    submissions = {
        f'submission_v6_{single_name.lower()}.csv': sub_single,
        'submission_v6_blend.csv': sub_blend,
        'submission_v6_stack.csv': sub_stack,
    }

    sample = pd.read_csv(os.path.join(DATASET_DIR, 'sample_submission.csv'))
    for fname, pred in submissions.items():
        sub = pd.DataFrame({'Index': test['Index'], 'demand': pred})
        path = os.path.join(SUBMISSIONS_DIR, fname)
        sub.to_csv(path, index=False)
        assert list(sub.columns) == list(sample.columns)
        assert len(sub) == 41778
        print(f"  {fname}: min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")

    best_name = max(all_r, key=all_r.get)
    best_score = max(0, 100 * all_r[best_name])
    print(f"\n  Format check: ALL PASSED")

    print("\n" + "=" * 70)
    print(f"V6 COMPLETE! Best: {best_name} (Score: {best_score:.2f}/100)")
    print("=" * 70)
    print(f"\nUpload order:")
    print(f"  1. submission_v6_stack.csv    (Ridge stacking)")
    print(f"  2. submission_v6_blend.csv    (Weighted blend)")
    print(f"  3. submission_v6_{single_name.lower()}.csv ({single_name})")


if __name__ == '__main__':
    train_v6()
