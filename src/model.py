"""
=============================================================
  MODULE 3: Train Regression Models for Risk Prediction
=============================================================
  Trains 4 models (Linear, Polynomial, Random Forest, Gradient Boosting),
  compares them, saves the best one.

  Maps to Practical 3 (Regression) + Practical 5 (Ensemble Learning)
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os


def train_models(data_path, model_dir):
    """Train and compare regression models for accident risk prediction."""

    # ---- 1. Load Data ----
    df = pd.read_csv(data_path)
    print(f"[DATA] Loaded {len(df)} samples from {data_path}")

    # ---- 2. Encode Categorical Features ----
    le_road = LabelEncoder()
    le_weather = LabelEncoder()
    df['road_type_enc'] = le_road.fit_transform(df['road_type'])
    df['weather_enc'] = le_weather.fit_transform(df['weather'])

    # ---- 3. Feature Selection ----
    feature_cols = [
        'road_type_enc', 'hour_of_day', 'day_of_week', 'weather_enc',
        'traffic_density', 'speed_limit', 'has_signal', 'num_lanes'
    ]
    X = df[feature_cols]
    y = df['accident_risk']

    # ---- 4. Train-Test Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # ---- 5. Define Models ----
    # P3: Linear + Polynomial regression
    # P5: Random Forest (Bagging) + Gradient Boosting
    models = {
        'Linear Regression':     LinearRegression(),
        'Polynomial (degree 2)': make_pipeline(PolynomialFeatures(2), LinearRegression()),
        'Random Forest':         RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting':     GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    # ---- 6. Train and Compare ----
    print(f"\n{'=' * 60}")
    print(f"  {'Model':<25} {'MSE':>8} {'RMSE':>8} {'R²':>8}")
    print(f"{'=' * 60}")

    best_model_name = None
    best_model_obj = None
    best_r2 = -999

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        results[name] = {'mse': mse, 'rmse': rmse, 'r2': r2}
        print(f"  {name:<25} {mse:>8.4f} {rmse:>8.4f} {r2:>8.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model_obj = model

    print(f"{'=' * 60}")
    print(f"\n>> Best Model: {best_model_name} (R2 = {best_r2:.4f})")

    # ---- 7. Save Best Model + Encoders ----
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(best_model_obj, os.path.join(model_dir, 'risk_model.pkl'))
    joblib.dump(le_road, os.path.join(model_dir, 'le_road.pkl'))
    joblib.dump(le_weather, os.path.join(model_dir, 'le_weather.pkl'))

    print(f"\n[OK] Saved to {model_dir}/:")
    print(f"   - risk_model.pkl ({best_model_name})")
    print(f"   - le_road.pkl")
    print(f"   - le_weather.pkl")

    return results, best_model_name


if __name__ == '__main__':
    # Paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'road_risk_data.csv')
    model_dir = os.path.join(project_root, 'models')

    if not os.path.exists(data_path):
        print("[ERROR] Dataset not found! Run 'python data/generate_data.py' first.")
    else:
        train_models(data_path, model_dir)
