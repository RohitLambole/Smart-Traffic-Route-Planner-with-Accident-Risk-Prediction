"""
Generates 3 visual graphs for the project:
  1. Model Comparison Bar Chart (MSE, RMSE, R2)
  2. Risk Distribution Histogram
  3. Feature Importance from Gradient Boosting
Run: python src/generate_graphs.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ── paths ──────────────────────────────────────────────────────────────────
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path    = os.path.join(project_root, 'data', 'road_risk_data.csv')
model_dir    = os.path.join(project_root, 'models')
out_dir      = project_root          # save PNGs next to main.py

plt.rcParams.update({'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#16213e',
                     'axes.labelcolor': 'white', 'xtick.color': 'white',
                     'ytick.color': 'white', 'text.color': 'white',
                     'axes.titlecolor': 'white', 'grid.color': '#333355'})

# ── load & encode data ──────────────────────────────────────────────────────
df = pd.read_csv(data_path)
le_road    = LabelEncoder().fit(df['road_type'])
le_weather = LabelEncoder().fit(df['weather'])
df['road_type_enc'] = le_road.transform(df['road_type'])
df['weather_enc']   = le_weather.transform(df['weather'])

features = ['road_type_enc','hour_of_day','day_of_week','weather_enc',
            'traffic_density','speed_limit','has_signal','num_lanes']
X = df[features]
y = df['accident_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── train 4 models ─────────────────────────────────────────────────────────
models = {
    'Linear\nRegression': LinearRegression(),
    'Polynomial\n(deg=2)': Pipeline([('poly', PolynomialFeatures(degree=2)),
                                     ('lr',   LinearRegression())]),
    'Random\nForest':      RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient\nBoosting':  GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    mse  = mean_squared_error(y_test, pred)
    r2   = r2_score(y_test, pred)
    results[name] = {'MSE': round(mse,4), 'RMSE': round(np.sqrt(mse),4), 'R2': round(r2,4)}

names  = list(results.keys())
mse_v  = [results[n]['MSE']  for n in names]
rmse_v = [results[n]['RMSE'] for n in names]
r2_v   = [results[n]['R2']   for n in names]

# ════════════════════════════════════════════════════════════════════════════
# GRAPH 1 — Model Comparison (3 sub-plots)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']   # red→orange→blue→green

for ax, values, label, best_low in zip(
        axes,
        [mse_v, rmse_v, r2_v],
        ['MSE (lower is better)', 'RMSE (lower is better)', 'R² Score (higher is better)'],
        [True, True, False]):

    bars = ax.bar(names, values, color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    ax.set_title(label, fontsize=11, pad=10)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top','right','left','bottom']].set_visible(False)

    # annotate bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # highlight best bar
    best_idx = values.index(min(values) if best_low else max(values))
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(2.5)
    ax.text(bars[best_idx].get_x() + bars[best_idx].get_width()/2,
            -max(values)*0.08, 'BEST', ha='center', color='gold', fontsize=9, fontweight='bold')

plt.tight_layout()
p1 = os.path.join(out_dir, 'graph_model_comparison.png')
plt.savefig(p1, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print(f'[OK] Saved: graph_model_comparison.png')

# ════════════════════════════════════════════════════════════════════════════
# GRAPH 2 — Risk Score Distribution by Road Type
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Accident Risk Distribution Analysis', fontsize=15, fontweight='bold')

# left: histogram
ax = axes[0]
ax.hist(df['accident_risk'], bins=30, color='#3498db', edgecolor='#1a1a2e', alpha=0.85)
ax.axvline(0.7, color='#e74c3c', linewidth=2, linestyle='--', label='Danger Threshold (0.7)')
ax.axvline(df['accident_risk'].mean(), color='#f39c12', linewidth=2,
           linestyle='--', label=f'Mean ({df["accident_risk"].mean():.2f})')
ax.set_xlabel('Accident Risk Score')
ax.set_ylabel('Number of Road Samples')
ax.set_title('Overall Risk Distribution (800 samples)')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# right: boxplot per road type
ax = axes[1]
road_types = ['highway', 'main_road', 'residential', 'lane']
type_colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
data_by_type = [df[df['road_type'] == rt]['accident_risk'].values for rt in road_types]
bp = ax.boxplot(data_by_type, patch_artist=True, widths=0.5,
                medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], type_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
for element in ['whiskers','caps','fliers']:
    for item in bp[element]:
        item.set_color('white')
ax.set_xticklabels(['Highway', 'Main Road', 'Residential', 'Lane'])
ax.set_ylabel('Accident Risk Score')
ax.set_title('Risk by Road Type')
ax.axhline(0.7, color='#e74c3c', linewidth=1.5, linestyle='--', alpha=0.7)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
p2 = os.path.join(out_dir, 'graph_risk_distribution.png')
plt.savefig(p2, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print(f'[OK] Saved: graph_risk_distribution.png')

# ════════════════════════════════════════════════════════════════════════════
# GRAPH 3 — Feature Importance (Gradient Boosting)
# ════════════════════════════════════════════════════════════════════════════
gb_model = joblib.load(os.path.join(model_dir, 'risk_model.pkl'))
feat_names  = ['Road Type', 'Hour of Day', 'Day of Week', 'Weather',
               'Traffic Density', 'Speed Limit', 'Has Signal', 'Num Lanes']
importances = gb_model.feature_importances_
sorted_idx  = np.argsort(importances)

fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = ['#2ecc71' if i < 4 else '#e74c3c' for i in range(len(importances))]
bar_colors_sorted = [bar_colors[i] for i in sorted_idx]
bars = ax.barh([feat_names[i] for i in sorted_idx], importances[sorted_idx],
               color=bar_colors_sorted, edgecolor='white', linewidth=0.4)
for bar, val in zip(bars, importances[sorted_idx]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
ax.set_xlabel('Feature Importance Score')
ax.set_title('Feature Importance — Gradient Boosting Model\n(which inputs affect accident risk most?)',
             fontsize=12, fontweight='bold')
ax.set_xlim(0, max(importances) * 1.2)
ax.grid(axis='x', alpha=0.3)
ax.spines[['top','right','bottom']].set_visible(False)
legend = [mpatches.Patch(color='#e74c3c', label='High importance'),
          mpatches.Patch(color='#2ecc71', label='Lower importance')]
ax.legend(handles=legend, loc='lower right')

plt.tight_layout()
p3 = os.path.join(out_dir, 'graph_feature_importance.png')
plt.savefig(p3, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print(f'[OK] Saved: graph_feature_importance.png')

print('\n[DONE] All 3 graphs generated:')
print(f'  1. graph_model_comparison.png')
print(f'  2. graph_risk_distribution.png')
print(f'  3. graph_feature_importance.png')
print('Open them with: start graph_model_comparison.png')
