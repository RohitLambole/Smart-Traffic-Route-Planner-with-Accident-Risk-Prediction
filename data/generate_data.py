"""
=============================================================
  STEP 1: Generate Synthetic Road Risk Dataset
=============================================================
  WHY: We need training data for our regression model.
  Synthetic data lets us control patterns and skip cleaning.

  WHAT IT CREATES: road_risk_data.csv with 800 road samples,
  each with features like road_type, weather, time, etc.
  and a target: accident_risk (0 to 1).
=============================================================
"""

import pandas as pd
import numpy as np
import os

def generate_road_data(n_samples=800, seed=42):
    """Generate synthetic road risk data with realistic patterns."""
    np.random.seed(seed)

    # ---- Raw features ----
    data = {
        'road_type':       np.random.choice(['highway', 'main_road', 'residential', 'lane'], n_samples),
        'hour_of_day':     np.random.randint(0, 24, n_samples),
        'day_of_week':     np.random.randint(0, 7, n_samples),
        'weather':         np.random.choice(['clear', 'rain', 'fog'], n_samples, p=[0.5, 0.3, 0.2]),
        'traffic_density': np.round(np.random.uniform(0.1, 1.0, n_samples), 2),
        'speed_limit':     np.random.choice([20, 30, 40, 60, 80], n_samples),
        'has_signal':      np.random.choice([0, 1], n_samples),
        'num_lanes':       np.random.choice([1, 2, 4], n_samples),
    }

    df = pd.DataFrame(data)

    # ---- Build risk score from logical rules ----
    # (This is the "ground truth" the model will try to learn)
    risk = np.zeros(n_samples)

    # Rule 1: Higher speed limit → higher risk
    risk += (df['speed_limit'] / 80) * 0.25

    # Rule 2: Bad weather → higher risk
    risk += df['weather'].map({'clear': 0.0, 'rain': 0.20, 'fog': 0.25}).values

    # Rule 3: Night time (10 PM – 5 AM) → higher risk
    is_night = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 5)).astype(float)
    risk += is_night * 0.15

    # Rule 4: More traffic → higher risk
    risk += df['traffic_density'] * 0.15

    # Rule 5: No traffic signal → higher risk
    risk += (1 - df['has_signal']) * 0.10

    # Rule 6: Road type base risk
    risk += df['road_type'].map({
        'highway': 0.10,
        'main_road': 0.05,
        'residential': 0.00,
        'lane': -0.05
    }).values

    # Rule 7: Interaction — rain + highway + night = extra dangerous
    interaction = ((df['weather'] == 'rain') &
                   (df['road_type'] == 'highway') &
                   is_night.astype(bool)).astype(float)
    risk += interaction * 0.15

    # Add realistic noise (real world isn't perfectly predictable)
    risk += np.random.normal(0, 0.04, n_samples)

    # Clamp to valid range [0, 1]
    df['accident_risk'] = np.clip(risk, 0.0, 1.0).round(3)

    return df


if __name__ == '__main__':
    print("Generating synthetic road risk data...")
    df = generate_road_data()

    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), 'road_risk_data.csv')
    df.to_csv(output_path, index=False)

    print(f"[OK] Saved {len(df)} samples to: {output_path}")
    print(f"\n--- Sample Data ---")
    print(df.head(10).to_string(index=False))
    print(f"\n--- Risk Statistics ---")
    print(f"   Mean:  {df['accident_risk'].mean():.3f}")
    print(f"   Min:   {df['accident_risk'].min():.3f}")
    print(f"   Max:   {df['accident_risk'].max():.3f}")
    print(f"   Std:   {df['accident_risk'].std():.3f}")
