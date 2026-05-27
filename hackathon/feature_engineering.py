"""
feature_engineering.py — Data Loading, Cleaning & Feature Engineering
Flipkart Gridlock Hackathon 2.0

Loads train.csv and test.csv, handles missing values, and engineers
15+ features for the traffic demand prediction model.
"""

import pandas as pd
import numpy as np
import math
import os
import sys

# Add hackathon dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import decode_geohash_batch, parse_timestamp, time_slot_index, cyclic_encode


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'dataset', 'train.csv')
TEST_PATH = os.path.join(BASE_DIR, 'dataset', 'test.csv')


# ============================================================
# CORE FEATURE ENGINEERING FUNCTION
# ============================================================

def engineer_features(df, geo_stats=None, is_train=True):
    """
    Apply all feature engineering transformations to a DataFrame.
    
    Args:
        df: Raw DataFrame loaded from CSV
        geo_stats: Dict of geohash-level statistics (computed from train, applied to both)
        is_train: If True, compute geo_stats from this data. If False, use provided geo_stats.
    
    Returns:
        tuple: (engineered_df, geo_stats_dict)
    """
    df = df.copy()
    print(f"  Starting with {len(df)} rows, {len(df.columns)} columns")
    
    # ----------------------------------------------------------
    # 1. TIMESTAMP FEATURES
    # ----------------------------------------------------------
    print("  Engineering timestamp features...")
    
    # Parse hour and minute
    parsed = df['timestamp'].apply(lambda ts: parse_timestamp(ts))
    df['hour'] = parsed.apply(lambda x: x[0])
    df['minute'] = parsed.apply(lambda x: x[1])
    
    # Time slot index (0-95)
    df['time_slot'] = df.apply(lambda row: time_slot_index(row['hour'], row['minute']), axis=1)
    
    # Cyclic encoding of hour (so 23:00 is close to 0:00)
    cyclic = df['hour'].apply(lambda h: cyclic_encode(h, 24))
    df['sin_hour'] = cyclic.apply(lambda x: x[0])
    df['cos_hour'] = cyclic.apply(lambda x: x[1])
    
    # Cyclic encoding of time_slot
    cyclic_slot = df['time_slot'].apply(lambda s: cyclic_encode(s, 96))
    df['sin_slot'] = cyclic_slot.apply(lambda x: x[0])
    df['cos_slot'] = cyclic_slot.apply(lambda x: x[1])
    
    # Rush hour flag (morning 8-10, evening 5-8)
    df['is_rush_hour'] = df['hour'].apply(lambda h: 1 if h in [8, 9, 10, 17, 18, 19] else 0)
    
    # Night flag (11 PM - 5 AM)
    df['is_night'] = df['hour'].apply(lambda h: 1 if h in [23, 0, 1, 2, 3, 4, 5] else 0)
    
    # Morning/Afternoon/Evening/Night period
    def get_period(h):
        if 6 <= h < 12:
            return 0  # Morning
        elif 12 <= h < 17:
            return 1  # Afternoon
        elif 17 <= h < 22:
            return 2  # Evening
        else:
            return 3  # Night
    df['period'] = df['hour'].apply(get_period)
    
    # ----------------------------------------------------------
    # 2. GEOHASH / SPATIAL FEATURES
    # ----------------------------------------------------------
    print("  Decoding geohashes to lat/lon...")
    
    lats, lons = decode_geohash_batch(df['geohash'])
    df['latitude'] = lats
    df['longitude'] = lons
    
    # Geohash prefix clustering (area-level grouping)
    df['geo_prefix_4'] = df['geohash'].str[:4]
    df['geo_prefix_5'] = df['geohash'].str[:5]
    
    # ----------------------------------------------------------
    # 3. GEOHASH-LEVEL AGGREGATE STATS (from training data)
    # ----------------------------------------------------------
    if is_train and 'demand' in df.columns:
        print("  Computing geohash-level demand statistics from training data...")
        
        # Stats from Day 48 only (historical baseline)
        day48 = df[df['day'] == 48]
        
        geo_stats = {}
        
        # Per-geohash demand mean & std
        geo_demand = day48.groupby('geohash')['demand'].agg(['mean', 'std', 'median', 'max']).reset_index()
        geo_demand.columns = ['geohash', 'geo_demand_mean', 'geo_demand_std', 'geo_demand_median', 'geo_demand_max']
        geo_demand['geo_demand_std'] = geo_demand['geo_demand_std'].fillna(0)
        geo_stats['geo_demand'] = geo_demand
        
        # Per-geohash + time_slot demand mean (time-of-day pattern per zone)
        geo_time = day48.groupby(['geohash', 'time_slot'])['demand'].mean().reset_index()
        geo_time.columns = ['geohash', 'time_slot', 'geo_time_demand_mean']
        geo_stats['geo_time_demand'] = geo_time
        
        # Per geo_prefix_4 demand mean (area-level baseline)
        prefix_demand = day48.groupby('geo_prefix_4')['demand'].mean().reset_index()
        prefix_demand.columns = ['geo_prefix_4', 'prefix4_demand_mean']
        geo_stats['prefix4_demand'] = prefix_demand
        
        # Global fallback mean
        geo_stats['global_mean'] = day48['demand'].mean()
        geo_stats['global_median'] = day48['demand'].median()
    
    # Merge geohash stats
    if geo_stats is not None:
        print("  Merging geohash-level statistics...")
        
        # Per-geohash stats
        df = df.merge(geo_stats['geo_demand'], on='geohash', how='left')
        df['geo_demand_mean'] = df['geo_demand_mean'].fillna(geo_stats['global_mean'])
        df['geo_demand_std'] = df['geo_demand_std'].fillna(0)
        df['geo_demand_median'] = df['geo_demand_median'].fillna(geo_stats['global_median'])
        df['geo_demand_max'] = df['geo_demand_max'].fillna(geo_stats['global_mean'])
        
        # Per-geohash + time_slot stats
        df = df.merge(geo_stats['geo_time_demand'], on=['geohash', 'time_slot'], how='left')
        df['geo_time_demand_mean'] = df['geo_time_demand_mean'].fillna(df['geo_demand_mean'])
        
        # Per-prefix stats
        df = df.merge(geo_stats['prefix4_demand'], on='geo_prefix_4', how='left')
        df['prefix4_demand_mean'] = df['prefix4_demand_mean'].fillna(geo_stats['global_mean'])
    
    # ----------------------------------------------------------
    # 4. ROAD & ENVIRONMENT FEATURES (Encoding)
    # ----------------------------------------------------------
    print("  Encoding categorical features...")
    
    # RoadType — fill missing, then encode
    df['RoadType'] = df['RoadType'].fillna('Unknown')
    road_type_map = {'Street': 0, 'Residential': 1, 'Highway': 2, 'Unknown': -1}
    df['road_type_encoded'] = df['RoadType'].map(road_type_map).fillna(-1).astype(int)
    
    # Weather — fill missing, then encode
    df['Weather'] = df['Weather'].fillna('Unknown')
    weather_map = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3, 'Unknown': -1}
    df['weather_encoded'] = df['Weather'].map(weather_map).fillna(-1).astype(int)
    
    # LargeVehicles — binary
    df['large_vehicles'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    
    # Landmarks — binary
    df['landmarks'] = (df['Landmarks'] == 'Yes').astype(int)
    
    # Temperature — fill missing with median
    temp_median = df['Temperature'].median()
    df['temperature'] = df['Temperature'].fillna(temp_median)
    
    # Temperature bins
    df['temp_bin'] = pd.cut(df['temperature'], bins=[-20, 5, 15, 25, 35, 50], labels=[0, 1, 2, 3, 4])
    df['temp_bin'] = df['temp_bin'].cat.add_categories(-1).fillna(-1).astype(int)
    
    # ----------------------------------------------------------
    # 5. INTERACTION FEATURES
    # ----------------------------------------------------------
    print("  Creating interaction features...")
    
    # Road capacity proxy: lanes × large_vehicles_allowed
    df['road_capacity'] = df['NumberofLanes'] * (1 + df['large_vehicles'])
    
    # Weather severity score (higher = worse driving conditions)
    weather_severity = {'Sunny': 0, 'Rainy': 2, 'Foggy': 3, 'Snowy': 4, 'Unknown': 1}
    df['weather_severity'] = df['Weather'].map(weather_severity).fillna(1)
    
    # Rush hour × road type interaction
    df['rush_road_interaction'] = df['is_rush_hour'] * df['road_type_encoded']
    
    # Landmark × rush hour
    df['landmark_rush'] = df['landmarks'] * df['is_rush_hour']
    
    print(f"  Done! Final shape: {df.shape}")
    
    return df, geo_stats


# ============================================================
# SELECT FEATURES FOR MODEL
# ============================================================

FEATURE_COLUMNS = [
    # Temporal
    'hour', 'minute', 'time_slot', 'sin_hour', 'cos_hour',
    'sin_slot', 'cos_slot', 'is_rush_hour', 'is_night', 'period',
    # Spatial
    'latitude', 'longitude',
    # Geohash aggregate
    'geo_demand_mean', 'geo_demand_std', 'geo_demand_median', 'geo_demand_max',
    'geo_time_demand_mean', 'prefix4_demand_mean',
    # Road
    'road_type_encoded', 'NumberofLanes', 'large_vehicles', 'landmarks',
    'road_capacity',
    # Environment
    'temperature', 'temp_bin', 'weather_encoded', 'weather_severity',
    # Interactions
    'rush_road_interaction', 'landmark_rush',
    # Day
    'day',
]

TARGET_COLUMN = 'demand'


# ============================================================
# MAIN: LOAD, ENGINEER, SAVE
# ============================================================

def load_and_engineer():
    """
    Full pipeline: Load train + test CSVs, engineer features, return ready data.
    
    Returns:
        tuple: (train_df, test_df, feature_columns, geo_stats)
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"Train: {train.shape}")
    print(f"Test:  {test.shape}")
    
    print("\n" + "=" * 60)
    print("ENGINEERING TRAIN FEATURES")
    print("=" * 60)
    train_eng, geo_stats = engineer_features(train, is_train=True)
    
    print("\n" + "=" * 60)
    print("ENGINEERING TEST FEATURES")
    print("=" * 60)
    test_eng, _ = engineer_features(test, geo_stats=geo_stats, is_train=False)
    
    # Verify all feature columns exist
    missing_train = [c for c in FEATURE_COLUMNS if c not in train_eng.columns]
    missing_test = [c for c in FEATURE_COLUMNS if c not in test_eng.columns]
    
    if missing_train:
        print(f"[WARN] Missing train features: {missing_train}")
    if missing_test:
        print(f"[WARN] Missing test features: {missing_test}")
    
    print(f"[OK] Feature engineering complete!")
    print(f"   Train: {train_eng.shape}, Test: {test_eng.shape}")
    print(f"   Features: {len(FEATURE_COLUMNS)} columns")
    
    return train_eng, test_eng, FEATURE_COLUMNS, geo_stats


if __name__ == '__main__':
    train_df, test_df, features, geo_stats = load_and_engineer()
    
    print("\n" + "=" * 60)
    print("FEATURE PREVIEW")
    print("=" * 60)
    print(f"\nFeature columns ({len(features)}):")
    for i, f in enumerate(features, 1):
        print(f"  {i:2d}. {f}")
    
    print(f"\nTrain target stats:")
    print(f"  Mean:   {train_df[TARGET_COLUMN].mean():.6f}")
    print(f"  Median: {train_df[TARGET_COLUMN].median():.6f}")
    print(f"  Std:    {train_df[TARGET_COLUMN].std():.6f}")
    
    print(f"\nSample engineered row:")
    print(train_df[features].iloc[0])
