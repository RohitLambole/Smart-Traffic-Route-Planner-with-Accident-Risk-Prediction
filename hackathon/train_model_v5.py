"""
train_model_v5.py - COMPLETE RETHINK
Flipkart Gridlock Hackathon 2.0

Insight from leaderboard: Top teams score 91-93 (R2=0.91-0.93).
Our V3 scores ~61. The gap means we're not capturing the core pattern.

Key realization: Traffic demand at a geohash is HIGHLY repetitive.
Day 49 demand ≈ Day 48 demand at the same (geohash, time_slot).

Strategy:
  1. Pure lookup baseline: predict Day 49 demand = Day 48 demand at same (geohash, time_slot)
  2. Target-encode geohash (don't just use prefix or lat/lon)
  3. Train model to predict RESIDUALS on top of the lookup
  4. Use CatBoost/LightGBM with native categorical handling for geohash
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
from utils import decode_geohash_batch, parse_timestamp, time_slot_index, cyclic_encode

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'submissions')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


def load_data():
    train = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATASET_DIR, 'test.csv'))
    return train, test


def parse_time_features(df):
    """Parse timestamps into numeric features."""
    parsed = df['timestamp'].apply(lambda ts: parse_timestamp(ts))
    df['hour'] = parsed.apply(lambda x: x[0])
    df['minute'] = parsed.apply(lambda x: x[1])
    df['time_slot'] = df.apply(lambda r: time_slot_index(r['hour'], r['minute']), axis=1)
    return df


def train_v5():
    print("=" * 70)
    print("GRIDLOCK HACKATHON 2.0 - V5 (LOOKUP + RESIDUAL MODEL)")
    print("  Target: Score 90+ (current best: 61)")
    print("=" * 70)

    # ============================================================
    # STEP 1: Load and parse basic features
    # ============================================================
    print("\n[1/6] Loading data...")
    train, test = load_data()
    train = parse_time_features(train)
    test = parse_time_features(test)
    print(f"  Train: {train.shape}, Test: {test.shape}")

    # ============================================================
    # STEP 2: Build the LOOKUP TABLE from Day 48
    # ============================================================
    print("\n[2/6] Building lookup table from Day 48...")
    day48 = train[train['day'] == 48].copy()
    day49_train = train[train['day'] == 49].copy()

    # Primary lookup: exact (geohash, time_slot) from Day 48
    lookup_exact = day48.groupby(['geohash', 'time_slot'])['demand'].mean().reset_index()
    lookup_exact.columns = ['geohash', 'time_slot', 'lookup_demand']

    # Fallback 1: geohash-level mean from Day 48
    lookup_geo = day48.groupby('geohash')['demand'].mean().reset_index()
    lookup_geo.columns = ['geohash', 'lookup_geo_mean']

    # Fallback 2: time_slot global mean from Day 48
    lookup_time = day48.groupby('time_slot')['demand'].mean().reset_index()
    lookup_time.columns = ['time_slot', 'lookup_time_mean']

    # Global fallback
    global_mean = day48['demand'].mean()

    print(f"  Lookup table: {len(lookup_exact)} (geohash, time_slot) entries")
    print(f"  Unique geohashes: {lookup_exact['geohash'].nunique()}")

    # ============================================================
    # STEP 3: Test PURE LOOKUP on Day 49 train data
    # ============================================================
    print("\n[3/6] Testing pure lookup baseline on Day 49...")
    
    day49_eval = day49_train.merge(lookup_exact, on=['geohash', 'time_slot'], how='left')
    day49_eval = day49_eval.merge(lookup_geo, on='geohash', how='left')
    day49_eval = day49_eval.merge(lookup_time, on='time_slot', how='left')

    # Fill missing lookups with fallbacks
    day49_eval['lookup_demand'] = day49_eval['lookup_demand'].fillna(day49_eval['lookup_geo_mean'])
    day49_eval['lookup_demand'] = day49_eval['lookup_demand'].fillna(day49_eval['lookup_time_mean'])
    day49_eval['lookup_demand'] = day49_eval['lookup_demand'].fillna(global_mean)

    lookup_r2 = r2_score(day49_eval['demand'], day49_eval['lookup_demand'])
    lookup_rmse = np.sqrt(mean_squared_error(day49_eval['demand'], day49_eval['lookup_demand']))
    print(f"  PURE LOOKUP: R2={lookup_r2:.6f}, RMSE={lookup_rmse:.6f}, Score={max(0,100*lookup_r2):.2f}")

    # ============================================================
    # STEP 4: Build RICH features for the residual model
    # ============================================================
    print("\n[4/6] Engineering rich features for residual model...")

    def build_features(df, day48_data, lookup_exact_df, lookup_geo_df, lookup_time_df, g_mean):
        """Build a comprehensive feature set."""
        df = df.copy()

        # Merge lookup values
        df = df.merge(lookup_exact_df, on=['geohash', 'time_slot'], how='left')
        df = df.merge(lookup_geo_df, on='geohash', how='left')
        df = df.merge(lookup_time_df, on='time_slot', how='left')
        df['lookup_demand'] = df['lookup_demand'].fillna(df['lookup_geo_mean'])
        df['lookup_demand'] = df['lookup_demand'].fillna(df['lookup_time_mean'])
        df['lookup_demand'] = df['lookup_demand'].fillna(g_mean)
        df['lookup_geo_mean'] = df['lookup_geo_mean'].fillna(g_mean)
        df['lookup_time_mean'] = df['lookup_time_mean'].fillna(g_mean)

        # Geohash statistics from Day 48
        geo_stats = day48_data.groupby('geohash')['demand'].agg(
            ['mean', 'std', 'median', 'min', 'max', 'count']
        ).reset_index()
        geo_stats.columns = ['geohash', 'geo_mean', 'geo_std', 'geo_median',
                             'geo_min', 'geo_max', 'geo_count']
        geo_stats['geo_std'] = geo_stats['geo_std'].fillna(0)
        df = df.merge(geo_stats, on='geohash', how='left')
        for col in ['geo_mean', 'geo_median', 'geo_min', 'geo_max']:
            df[col] = df[col].fillna(g_mean)
        df['geo_std'] = df['geo_std'].fillna(0)
        df['geo_count'] = df['geo_count'].fillna(0)

        # Per (geohash, time_slot) std from Day 48
        geo_ts_std = day48_data.groupby(['geohash', 'time_slot'])['demand'].std().reset_index()
        geo_ts_std.columns = ['geohash', 'time_slot', 'geo_ts_std']
        geo_ts_std['geo_ts_std'] = geo_ts_std['geo_ts_std'].fillna(0)
        df = df.merge(geo_ts_std, on=['geohash', 'time_slot'], how='left')
        df['geo_ts_std'] = df['geo_ts_std'].fillna(0)

        # Neighboring time slot demands from Day 48
        for offset in [-1, 1, -2, 2, -4, 4]:
            col_name = f'lookup_ts_offset_{offset}'
            offset_df = lookup_exact_df.copy()
            offset_df['time_slot'] = offset_df['time_slot'] - offset  # shift
            offset_df = offset_df.rename(columns={'lookup_demand': col_name})
            df = df.merge(offset_df[['geohash', 'time_slot', col_name]],
                         on=['geohash', 'time_slot'], how='left')
            df[col_name] = df[col_name].fillna(df['lookup_demand'])

        # Temporal features
        cyclic = df['hour'].apply(lambda h: cyclic_encode(h, 24))
        df['sin_hour'] = cyclic.apply(lambda x: x[0])
        df['cos_hour'] = cyclic.apply(lambda x: x[1])
        cyclic_slot = df['time_slot'].apply(lambda s: cyclic_encode(s, 96))
        df['sin_slot'] = cyclic_slot.apply(lambda x: x[0])
        df['cos_slot'] = cyclic_slot.apply(lambda x: x[1])
        df['is_rush'] = df['hour'].apply(lambda h: 1 if h in [8,9,10,17,18,19] else 0)
        df['is_night'] = df['hour'].apply(lambda h: 1 if h in [23,0,1,2,3,4,5] else 0)

        # Road features
        df['RoadType'] = df['RoadType'].fillna('Unknown')
        road_map = {'Street': 0, 'Residential': 1, 'Highway': 2, 'Unknown': -1}
        df['road_encoded'] = df['RoadType'].map(road_map).fillna(-1).astype(int)

        df['Weather'] = df['Weather'].fillna('Unknown')
        weather_map = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3, 'Unknown': -1}
        df['weather_encoded'] = df['Weather'].map(weather_map).fillna(-1).astype(int)

        df['large_vehicles'] = (df['LargeVehicles'] == 'Allowed').astype(int)
        df['landmarks'] = (df['Landmarks'] == 'Yes').astype(int)
        df['temperature'] = df['Temperature'].fillna(df['Temperature'].median())

        # Ratio features
        df['demand_vs_geo'] = df['lookup_demand'] / (df['geo_mean'] + 1e-8)
        df['demand_vs_time'] = df['lookup_demand'] / (df['lookup_time_mean'] + 1e-8)
        df['geo_vs_global'] = df['geo_mean'] / (g_mean + 1e-8)

        # Geohash target encoding (label encode geohash into integer)
        # This lets the tree model learn per-geohash patterns directly
        geohash_codes = pd.Categorical(df['geohash']).codes
        df['geohash_code'] = geohash_codes

        # Geohash prefix
        df['geo_prefix_4'] = pd.Categorical(df['geohash'].str[:4]).codes
        df['geo_prefix_5'] = pd.Categorical(df['geohash'].str[:5]).codes

        # Spatial
        lats, lons = decode_geohash_batch(df['geohash'])
        df['latitude'] = lats
        df['longitude'] = lons

        return df

    # Build features for all data
    train_feat = build_features(train, day48, lookup_exact, lookup_geo, lookup_time, global_mean)
    test_feat = build_features(test, day48, lookup_exact, lookup_geo, lookup_time, global_mean)

    # Feature list
    feature_cols = [
        # LOOKUP (most important!)
        'lookup_demand', 'lookup_geo_mean', 'lookup_time_mean',
        # Neighboring time slots
        'lookup_ts_offset_-1', 'lookup_ts_offset_1',
        'lookup_ts_offset_-2', 'lookup_ts_offset_2',
        'lookup_ts_offset_-4', 'lookup_ts_offset_4',
        # Geohash stats
        'geo_mean', 'geo_std', 'geo_median', 'geo_min', 'geo_max', 'geo_count',
        'geo_ts_std',
        # Temporal
        'hour', 'minute', 'time_slot', 'sin_hour', 'cos_hour',
        'sin_slot', 'cos_slot', 'is_rush', 'is_night',
        # Road & env
        'road_encoded', 'weather_encoded', 'NumberofLanes',
        'large_vehicles', 'landmarks', 'temperature',
        # Ratios
        'demand_vs_geo', 'demand_vs_time', 'geo_vs_global',
        # Geohash encoding
        'geohash_code', 'geo_prefix_4', 'geo_prefix_5',
        # Spatial
        'latitude', 'longitude',
        # Day
        'day',
    ]

    # Verify all features exist
    missing = [c for c in feature_cols if c not in test_feat.columns]
    if missing:
        print(f"  [WARN] Missing: {missing}")
        feature_cols = [c for c in feature_cols if c in test_feat.columns and c in train_feat.columns]

    print(f"  Features: {len(feature_cols)}")

    # ============================================================
    # STEP 5: Train models
    # ============================================================
    print("\n[5/6] Training models...")

    # Split for validation
    val_mask = train_feat['day'] == 49
    X_train_split = train_feat.loc[~val_mask, feature_cols]
    y_train_split = train_feat.loc[~val_mask, 'demand']
    X_val = train_feat.loc[val_mask, feature_cols]
    y_val = train_feat.loc[val_mask, 'demand']
    print(f"  Train split: {X_train_split.shape}, Val: {X_val.shape}")

    # --- LightGBM with categorical features ---
    print(f"\n  Training LightGBM V5...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=5000,
        max_depth=-1,
        learning_rate=0.01,
        num_leaves=511,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_samples=5,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_train_split, y_train_split,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)]
    )
    lgb_pred = np.clip(lgb_model.predict(X_val), 0, 1)
    lgb_r2 = r2_score(y_val, lgb_pred)
    print(f"  LightGBM V5: R2={lgb_r2:.6f}, Score={max(0,100*lgb_r2):.2f}")

    # --- XGBoost ---
    print(f"\n  Training XGBoost V5...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=5000,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=3,
        reg_alpha=0.05,
        reg_lambda=1.0,
        gamma=0.01,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=200
    )
    xgb_model.fit(X_train_split, y_train_split, eval_set=[(X_val, y_val)], verbose=500)
    xgb_pred = np.clip(xgb_model.predict(X_val), 0, 1)
    xgb_r2 = r2_score(y_val, xgb_pred)
    print(f"  XGBoost V5: R2={xgb_r2:.6f}, Score={max(0,100*xgb_r2):.2f}")

    # --- Optimized Ensemble ---
    print(f"\n  Optimizing ensemble weights...")
    best_w, best_r2_ens = 0.5, -1
    for w in np.arange(0.0, 1.01, 0.01):
        blend = w * lgb_pred + (1 - w) * xgb_pred
        r2 = r2_score(y_val, blend)
        if r2 > best_r2_ens:
            best_r2_ens = r2
            best_w = w
    
    print(f"  Best weight: LGB={best_w:.2f}, XGB={1-best_w:.2f}")
    print(f"  Ensemble V5: R2={best_r2_ens:.6f}, Score={max(0,100*best_r2_ens):.2f}")

    # --- Also try: blend with pure lookup ---
    print(f"\n  Testing blend with pure lookup...")
    day49_lookup = train_feat.loc[val_mask, 'lookup_demand'].values
    
    best_w2, best_r2_hybrid = 0, -1
    for w in np.arange(0.0, 1.01, 0.01):
        hybrid = w * day49_lookup + (1 - w) * xgb_pred
        r2 = r2_score(y_val, hybrid)
        if r2 > best_r2_hybrid:
            best_r2_hybrid = r2
            best_w2 = w
    print(f"  Best hybrid weight: Lookup={best_w2:.2f}, XGB={1-best_w2:.2f}")
    print(f"  Hybrid (Lookup+XGB): R2={best_r2_hybrid:.6f}, Score={max(0,100*best_r2_hybrid):.2f}")

    # Results summary
    print("\n" + "=" * 70)
    print("V5 RESULTS COMPARISON")
    print("=" * 70)
    all_results = {
        'Pure Lookup': lookup_r2,
        'LightGBM V5': lgb_r2,
        'XGBoost V5': xgb_r2,
        'Ensemble (LGB+XGB)': best_r2_ens,
        'Hybrid (Lookup+XGB)': best_r2_hybrid,
    }
    for name, r2 in sorted(all_results.items(), key=lambda x: -x[1]):
        print(f"  {name:<25} R2={r2:.6f}  Score={max(0,100*r2):.2f}")

    # ============================================================
    # STEP 6: Generate submissions (train on ALL data)
    # ============================================================
    print("\n[6/6] Training final models on ALL data & generating submissions...")

    X_all = train_feat[feature_cols]
    y_all = train_feat['demand']
    X_test_feat = test_feat[feature_cols]

    # Retrain LightGBM on all data
    best_lgb_iter = lgb_model.best_iteration_ if hasattr(lgb_model, 'best_iteration_') else 500
    lgb_final = lgb.LGBMRegressor(
        n_estimators=best_lgb_iter + 50,
        max_depth=-1, learning_rate=0.01, num_leaves=511,
        subsample=0.7, colsample_bytree=0.6, min_child_samples=5,
        reg_alpha=0.05, reg_lambda=0.5, random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_final.fit(X_all, y_all)

    # Retrain XGBoost on all data
    best_xgb_iter = xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else 500
    xgb_final = xgb.XGBRegressor(
        n_estimators=best_xgb_iter + 50,
        max_depth=8, learning_rate=0.01, subsample=0.7, colsample_bytree=0.6,
        min_child_weight=3, reg_alpha=0.05, reg_lambda=1.0, gamma=0.01,
        random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_final.fit(X_all, y_all)

    # Generate predictions
    lgb_test_pred = np.clip(lgb_final.predict(X_test_feat), 0, 1)
    xgb_test_pred = np.clip(xgb_final.predict(X_test_feat), 0, 1)
    lookup_test_pred = test_feat['lookup_demand'].values

    # Submission A: Best single model (XGBoost or LightGBM)
    if xgb_r2 > lgb_r2:
        best_single = xgb_test_pred
        best_single_name = "XGBoost"
    else:
        best_single = lgb_test_pred
        best_single_name = "LightGBM"

    sub_a = pd.DataFrame({'Index': test_feat['Index'], 'demand': best_single})
    path_a = os.path.join(SUBMISSIONS_DIR, 'submission_v5_best_single.csv')
    sub_a.to_csv(path_a, index=False)
    print(f"\n  [A] {best_single_name}: {path_a}")

    # Submission B: Ensemble
    ens_pred = best_w * lgb_test_pred + (1 - best_w) * xgb_test_pred
    ens_pred = np.clip(ens_pred, 0, 1)
    sub_b = pd.DataFrame({'Index': test_feat['Index'], 'demand': ens_pred})
    path_b = os.path.join(SUBMISSIONS_DIR, 'submission_v5_ensemble.csv')
    sub_b.to_csv(path_b, index=False)
    print(f"  [B] Ensemble: {path_b}")

    # Submission C: Hybrid (Lookup + Model)
    hybrid_pred = best_w2 * lookup_test_pred + (1 - best_w2) * xgb_test_pred
    hybrid_pred = np.clip(hybrid_pred, 0, 1)
    sub_c = pd.DataFrame({'Index': test_feat['Index'], 'demand': hybrid_pred})
    path_c = os.path.join(SUBMISSIONS_DIR, 'submission_v5_hybrid.csv')
    sub_c.to_csv(path_c, index=False)
    print(f"  [C] Hybrid: {path_c}")

    # Submission D: Pure lookup
    lookup_pred = np.clip(lookup_test_pred, 0, 1)
    sub_d = pd.DataFrame({'Index': test_feat['Index'], 'demand': lookup_pred})
    path_d = os.path.join(SUBMISSIONS_DIR, 'submission_v5_lookup.csv')
    sub_d.to_csv(path_d, index=False)
    print(f"  [D] Pure Lookup: {path_d}")

    # Verify all
    sample = pd.read_csv(os.path.join(DATASET_DIR, 'sample_submission.csv'))
    for name, sub in [('A', sub_a), ('B', sub_b), ('C', sub_c), ('D', sub_d)]:
        assert list(sub.columns) == list(sample.columns), f"{name}: columns!"
        assert len(sub) == 41778, f"{name}: rows!"
    print(f"\n  Format check: ALL PASSED")

    # Save
    joblib.dump({'lgb': lgb_final, 'xgb': xgb_final, 'features': feature_cols,
                 'ens_weight': best_w, 'hybrid_weight': best_w2},
                os.path.join(MODELS_DIR, 'models_v5.pkl'))

    best_overall = max(all_results, key=all_results.get)
    best_score = max(0, 100 * all_results[best_overall])

    print("\n" + "=" * 70)
    print(f"V5 COMPLETE! Best approach: {best_overall} (Score: {best_score:.2f}/100)")
    print("=" * 70)
    print(f"\nRecommended upload order:")
    print(f"  1. submission_v5_hybrid.csv    (Lookup + XGB blend)")
    print(f"  2. submission_v5_best_single.csv ({best_single_name})")
    print(f"  3. submission_v5_ensemble.csv  (LGB + XGB)")
    print(f"  4. submission_v5_lookup.csv    (Pure lookup baseline)")


if __name__ == '__main__':
    train_v5()
