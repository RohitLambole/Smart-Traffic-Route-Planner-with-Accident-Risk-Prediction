"""
=============================================================
  MODULE 4: Risk-Aware Routing
=============================================================
  This module connects the regression model (Module 3) to
  the A* algorithm (Module 2).

  It provides:
    - predict_edge_risk()   → risk score for one road segment
    - risk_aware_weight()   → combined weight for A* cost function
    - set_conditions()      → update weather, time, traffic

  Maps to Practical 3 (Regression) connected to Practical 2 (A*)
=============================================================
"""

import joblib
import numpy as np
import pandas as pd
import os

# ---- Load trained model and encoders ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_model_dir = os.path.join(_project_root, 'models')

risk_model = None
le_road = None
le_weather = None


def load_model():
    """Load the trained risk model and encoders."""
    global risk_model, le_road, le_weather
    try:
        risk_model = joblib.load(os.path.join(_model_dir, 'risk_model.pkl'))
        le_road = joblib.load(os.path.join(_model_dir, 'le_road.pkl'))
        le_weather = joblib.load(os.path.join(_model_dir, 'le_weather.pkl'))
        return True
    except FileNotFoundError:
        print("[ERROR] Model files not found! Run model.py first.")
        return False


# ---- Current environmental conditions ----
# (In a real system, this would come from APIs)
current_conditions = {
    'hour_of_day': 23,       # 11 PM — late night
    'day_of_week': 5,        # Saturday
    'weather': 'rain',       # Rainy night
    'traffic_density': 0.7,  # Heavy traffic
}


def set_conditions(hour=None, day=None, weather=None, traffic=None):
    """Update current conditions (simulates real-time environment)."""
    if hour is not None:
        current_conditions['hour_of_day'] = hour
    if day is not None:
        current_conditions['day_of_week'] = day
    if weather is not None:
        current_conditions['weather'] = weather
    if traffic is not None:
        current_conditions['traffic_density'] = traffic


def predict_edge_risk(node_a, node_b, edge_metadata):
    """
    Predict accident risk for a road segment.

    Takes edge metadata (road_type, speed_limit, etc.) and combines
    it with current conditions (weather, time) to predict risk.

    Returns: float between 0.0 (safe) and 1.0 (dangerous)
    """
    if risk_model is None:
        load_model()
        if risk_model is None:
            return 0.5  # fallback

    feature_dict = {
        'road_type_enc':   [le_road.transform([edge_metadata['road_type']])[0]],
        'hour_of_day':     [current_conditions['hour_of_day']],
        'day_of_week':     [current_conditions['day_of_week']],
        'weather_enc':     [le_weather.transform([current_conditions['weather']])[0]],
        'traffic_density': [current_conditions['traffic_density']],
        'speed_limit':     [edge_metadata['speed_limit']],
        'has_signal':      [edge_metadata['has_signal']],
        'num_lanes':       [edge_metadata['num_lanes']],
    }
    df_input = pd.DataFrame(feature_dict)
    prediction = risk_model.predict(df_input)[0]
    return float(np.clip(prediction, 0.0, 1.0))


# ---- Cost Function Parameters ----
ALPHA = 0.6       # Weight for distance
BETA = 0.4        # Weight for safety
RISK_SCALE = 10   # Scale risk to be comparable with distance


def risk_aware_weight(node_a, node_b, base_distance, edge_meta_lookup=None):
    """
    Custom weight function for A*.

    weight = α × distance + β × (risk × scale)

    This is the BRIDGE between the regression model and A*.
    A* calls this function for every edge it evaluates.
    """
    if edge_meta_lookup is None:
        from graph import get_edge_metadata
        edge_meta_lookup = get_edge_metadata

    metadata = edge_meta_lookup(node_a, node_b)
    risk = predict_edge_risk(node_a, node_b, metadata)

    weight = ALPHA * base_distance + BETA * (risk * RISK_SCALE)
    return weight


def create_weight_fn(edge_meta_lookup):
    """
    Factory: creates a weight function with the edge metadata lookup bound.
    This is what we pass to A*.
    """
    def weight_fn(node_a, node_b, base_distance):
        return risk_aware_weight(node_a, node_b, base_distance, edge_meta_lookup)
    return weight_fn
