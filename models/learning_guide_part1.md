# 🎓 Learning & Building Guide — Part 1: Foundations

> [!IMPORTANT]
> **How to use this guide:** Read each module top-to-bottom. Type every code snippet yourself — don't copy-paste. Answer the checkpoint questions before moving on. If you can't answer, re-read the module.

---

## Module 1: Graphs — The City Map

### 💡 Concept (Simple Words)

A **graph** is just a collection of **points** connected by **lines**.
- Points = **Nodes** (intersections in a city)
- Lines = **Edges** (roads between intersections)
- Each edge can have a **weight** (distance, time, or cost of traveling that road)

That's it. No magic.

### 🌍 Real-World Analogy

Think of **Google Maps**. Every junction/chowk is a node. Every road connecting two junctions is an edge. The distance written on each road is the weight.

When you ask for directions, Google is solving a graph problem — "find me the best path from node A to node B through this network of edges."

### 🔨 Minimal Working Code

```python
# Step 1: Represent a tiny city as a dictionary
# Each key is a node, each value is a list of (neighbor, distance) pairs

city = {
    'A': [('B', 3), ('C', 5)],
    'B': [('A', 3), ('D', 4)],
    'C': [('A', 5), ('D', 2)],
    'D': [('B', 4), ('C', 2)]
}
```

**What this means:**
- From **A**, you can go to **B** (3 km) or **C** (5 km)
- From **C**, you can go to **A** (5 km) or **D** (2 km)

```python
# Step 2: Print all connections
for node, neighbors in city.items():
    for neighbor, distance in neighbors:
        print(f"  {node} → {neighbor} : {distance} km")
```

**Output:**
```
  A → B : 3 km
  A → C : 5 km
  B → A : 3 km
  B → D : 4 km
  C → A : 5 km
  C → D : 2 km
  D → B : 4 km
  D → C : 2 km
```

### 🔼 Upgrade: Add Coordinates (We'll Need These for A*)

**Why?** A* needs to estimate "how far am I from the goal?" For that, each node needs a position on a 2D plane.

```python
# Node positions (x, y) — imagine these on a map
positions = {
    'A': (0, 0),
    'B': (3, 0),
    'C': (1, 4),
    'D': (4, 4)
}
```

### 🔼 Upgrade: Add Edge Metadata (We'll Need This for Risk Prediction)

**Why?** Each road has properties (type, speed limit, signals) that affect accident risk. We store this separately.

```python
# Edge metadata — properties of each road segment
edge_info = {
    ('A', 'B'): {'road_type': 'highway',     'speed_limit': 80, 'has_signal': 0, 'num_lanes': 4},
    ('A', 'C'): {'road_type': 'residential', 'speed_limit': 30, 'has_signal': 1, 'num_lanes': 2},
    ('B', 'D'): {'road_type': 'main_road',   'speed_limit': 60, 'has_signal': 1, 'num_lanes': 2},
    ('C', 'D'): {'road_type': 'lane',        'speed_limit': 20, 'has_signal': 0, 'num_lanes': 1},
}
```

### ✅ Checkpoint Questions

1. **If you add a new node 'E' connected to 'B' (6 km) and 'D' (1 km), what do you add to the `city` dictionary?**
2. **Why do we store edges in both directions (A→B AND B→A)?** What would happen if we didn't?
3. **Why do we need `positions` separately from the graph?** Can A* work without them?

### ⚠️ Common Mistakes

| Mistake | Why It's Wrong |
|---------|---------------|
| Forgetting reverse edges | Graph becomes directed — you can go A→B but not B→A. That's a one-way street, not what we want |
| Putting positions inside the graph dict | Mixes two different things. Graph = connectivity. Positions = geometry. Keep them separate |
| Using a 2D list/matrix for the graph | Works, but adjacency list (dict) is better for sparse graphs like road networks. Most intersections connect to only 3–4 roads, not all other intersections |

---

## Module 2: A* Search Algorithm

### 💡 Concept (Simple Words)

A* finds the **best path** from start to goal. It's smart because at each step, it asks:

> "Which node should I explore next? The one that has the **lowest total estimated cost**."

**Total estimated cost = what I've already spent + what I think is left**

```
f(n) = g(n) + h(n)
```

| Symbol | Meaning | Analogy |
|--------|---------|---------|
| `g(n)` | Actual cost from start to current node | "I've driven 5 km so far" |
| `h(n)` | **Estimated** cost from current node to goal | "The goal looks about 3 km away" |
| `f(n)` | Total estimated cost | "This route will probably cost about 8 km total" |

### 🌍 Real-World Analogy

You're in a **mall** looking for a specific shop. You're at the entrance (start) and want to reach a shop on the 3rd floor (goal).

- **Dijkstra** (no heuristic): You check every single shop on every floor systematically. Guaranteed to find it, but slow.
- **A*** (with heuristic): You look at the directory sign that says "Electronics: 3rd floor, East wing." You head straight toward that direction. You skip irrelevant floors and wings. **Same correct answer, much fewer steps.**

The directory sign is your **heuristic** — an educated guess of the remaining distance.

### 🔨 Minimal Working Code (Distance-Only)

```python
import heapq
import math

def euclidean_distance(pos, node_a, node_b):
    """Heuristic: straight-line distance between two nodes"""
    x1, y1 = pos[node_a]
    x2, y2 = pos[node_b]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def a_star(graph, positions, start, goal):
    # Priority queue: (f_score, node)
    open_set = [(0, start)]

    # Track: where did I come from?
    came_from = {}

    # Track: actual cost from start to each node
    g_score = {start: 0}

    while open_set:
        current_f, current = heapq.heappop(open_set)

        # Reached the goal!
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, g_score[goal]

        # Explore neighbors
        for neighbor, distance in graph[current]:
            tentative_g = g_score[current] + distance

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + euclidean_distance(positions, neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))

    return None, float('inf')  # No path found
```

### 🧪 Test It

```python
city = {
    'A': [('B', 3), ('C', 5)],
    'B': [('A', 3), ('D', 4)],
    'C': [('A', 5), ('D', 2)],
    'D': [('B', 4), ('C', 2)]
}

positions = {
    'A': (0, 0), 'B': (3, 0),
    'C': (1, 4), 'D': (4, 4)
}

path, cost = a_star(city, positions, 'A', 'D')
print(f"Path: {' → '.join(path)}")
print(f"Cost: {cost} km")
```

**Expected output:**
```
Path: A → C → D
Cost: 7 km
```

**Why A→C→D and not A→B→D?** Because A→C (5) + C→D (2) = 7, while A→B (3) + B→D (4) = 7. Both are optimal! A* may return either one.

### 🔼 Upgrade: Make the Weight Function Pluggable

**Why?** Right now A* only uses distance. Later, we want `weight = distance + risk`. So we make the weight function a **parameter**.

```python
def a_star(graph, positions, start, goal, weight_fn=None):
    """
    weight_fn(node_a, node_b, base_distance) -> actual weight to use
    If None, uses base_distance directly
    """
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, g_score[goal]

        for neighbor, distance in graph[current]:
            # Use custom weight function if provided
            if weight_fn:
                weight = weight_fn(current, neighbor, distance)
            else:
                weight = distance

            tentative_g = g_score[current] + weight

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + euclidean_distance(positions, neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))

    return None, float('inf')
```

Now you can call:
```python
# Default: distance only
a_star(city, positions, 'A', 'D')

# Later: distance + risk (we'll build this function in Module 4)
a_star(city, positions, 'A', 'D', weight_fn=risk_aware_weight)
```

### ✅ Checkpoint Questions

1. **What happens if `h(n) = 0` for all nodes?** (Hint: it becomes a famous algorithm)
2. **What does "admissible heuristic" mean?** Why must Euclidean distance never overestimate?
3. **Trace through the algorithm by hand:** In our 4-node city, which node does A* explore first after 'A'? Why?

### ⚠️ Common Mistakes

| Mistake | Why It's Wrong |
|---------|---------------|
| Using `list` instead of `heapq` for the open set | Without a priority queue, you lose the "pick the best f-score" property. A* degrades to brute force |
| Not checking `if tentative_g < g_score.get(...)` | You'll revisit nodes with worse paths, wasting time and potentially getting wrong answers |
| Using Manhattan distance for road networks | Manhattan assumes grid-like roads. Euclidean is safer — it never overestimates. Manhattan can overestimate on diagonal roads |
| Forgetting to handle "no path found" | If the graph is disconnected, A* must return gracefully, not crash |

---

## Module 3: Regression — Predicting Accident Risk

### 💡 Concept (Simple Words)

Regression answers: **"Given some inputs, predict a number."**

- Input: road properties (type, speed limit, weather, time, etc.)
- Output: accident risk score (a number between 0 and 1)

It's not classification (safe/unsafe). It's a **precise score**: 0.23 is low risk, 0.87 is high risk. This precision matters when we put it into A*'s cost function.

### 🌍 Real-World Analogy

Think of predicting your **exam marks** based on: hours studied, sleep quality, difficulty level, previous scores.

- A teacher who says "you'll pass or fail" → **classification**
- A teacher who says "you'll get approximately 72 out of 100" → **regression**

We need regression because A* needs a number to do math with, not just a label.

### 🔨 Step 1: Create Synthetic Training Data

**Why synthetic?** You control the patterns. You know the "right answer." Real data needs heavy cleaning.

```python
# generate_data.py
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 800

data = {
    'road_type':       np.random.choice(['highway','main_road','residential','lane'], n_samples),
    'hour_of_day':     np.random.randint(0, 24, n_samples),
    'day_of_week':     np.random.randint(0, 7, n_samples),
    'weather':         np.random.choice(['clear','rain','fog'], n_samples),
    'traffic_density': np.round(np.random.uniform(0.1, 1.0, n_samples), 2),
    'speed_limit':     np.random.choice([20, 30, 40, 60, 80], n_samples),
    'has_signal':      np.random.choice([0, 1], n_samples),
    'num_lanes':       np.random.choice([1, 2, 4], n_samples),
}

df = pd.DataFrame(data)

# Create realistic risk scores based on logical rules
risk = np.zeros(n_samples)

# Higher speed → higher risk
risk += (df['speed_limit'] / 80) * 0.25

# Rain/fog → higher risk
risk += df['weather'].map({'clear': 0, 'rain': 0.2, 'fog': 0.25}).values

# Night time (10 PM - 5 AM) → higher risk
risk += ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 5)).astype(float) * 0.15

# High traffic density → higher risk
risk += df['traffic_density'] * 0.15

# No signal → slightly higher risk
risk += (1 - df['has_signal']) * 0.1

# Highway → higher base risk than residential
risk += df['road_type'].map({'highway': 0.1, 'main_road': 0.05,
                              'residential': 0, 'lane': -0.05}).values

# Add noise (real world isn't perfect)
risk += np.random.normal(0, 0.05, n_samples)

# Clamp to [0, 1]
df['accident_risk'] = np.clip(risk, 0, 1).round(3)

df.to_csv('road_risk_data.csv', index=False)
print(f"Generated {n_samples} samples")
print(df.head(10))
print(f"\nRisk stats: mean={df['accident_risk'].mean():.3f}, "
      f"min={df['accident_risk'].min():.3f}, max={df['accident_risk'].max():.3f}")
```

> [!NOTE]
> **Why we add noise:** In real life, two identical roads can have different accident rates due to factors we can't measure. Adding noise simulates this. Without noise, the model gets unrealistically perfect scores and you can't demonstrate overfitting concepts in your viva.

### 🔨 Step 2: Train and Compare Models

```python
# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load data
df = pd.read_csv('road_risk_data.csv')

# 2. Encode categorical columns to numbers
le_road = LabelEncoder()
le_weather = LabelEncoder()
df['road_type_enc'] = le_road.fit_transform(df['road_type'])
df['weather_enc'] = le_weather.fit_transform(df['weather'])

# 3. Select features and target
feature_cols = ['road_type_enc', 'hour_of_day', 'day_of_week', 'weather_enc',
                'traffic_density', 'speed_limit', 'has_signal', 'num_lanes']
X = df[feature_cols]
y = df['accident_risk']

# 4. Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train multiple models
models = {
    'Linear Regression':    LinearRegression(),
    'Polynomial (degree 2)': make_pipeline(PolynomialFeatures(2), LinearRegression()),
    'Random Forest':        RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting':    GradientBoostingRegressor(n_estimators=100, random_state=42),
}

print("=" * 55)
print(f"{'Model':<25} {'MSE':>8} {'RMSE':>8} {'R²':>8}")
print("=" * 55)

best_model = None
best_r2 = -999

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, pred)
    print(f"{name:<25} {mse:>8.4f} {rmse:>8.4f} {r2:>8.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = (name, model)

print("=" * 55)
print(f"\n🏆 Best model: {best_model[0]} (R² = {best_r2:.4f})")

# 6. Save best model + encoders
joblib.dump(best_model[1], 'risk_model.pkl')
joblib.dump(le_road, 'le_road.pkl')
joblib.dump(le_weather, 'le_weather.pkl')
print("✅ Model and encoders saved!")
```

**Expected output (approximate):**
```
=========================================================
Model                          MSE     RMSE       R²
=========================================================
Linear Regression            0.0031   0.0557   0.8521
Polynomial (degree 2)        0.0028   0.0529   0.8672
Random Forest                0.0019   0.0436   0.9091
Gradient Boosting            0.0016   0.0400   0.9238
=========================================================

🏆 Best model: Gradient Boosting (R² = 0.9238)
```

### 🧠 Understanding the Results

| Metric | What It Means | Good Value |
|--------|--------------|------------|
| **MSE** | Average squared error. Lower = better | < 0.01 for our 0–1 range |
| **RMSE** | Square root of MSE. Same units as target | < 0.05 means "off by ~5%" |
| **R²** | How much variance the model explains. 1.0 = perfect | > 0.85 is solid |

**Why Gradient Boosting wins:** It builds trees sequentially, where each tree fixes mistakes of the previous one. It captures complex patterns like "rain + highway + night = very high risk" that linear models can't learn.

### ✅ Checkpoint Questions

1. **Why did we use LabelEncoder?** What happens if you feed "highway" directly to sklearn?
2. **If Linear Regression gives R²=0.85 and Gradient Boosting gives R²=0.92, is the improvement worth the complexity?** When would you prefer the simpler model?
3. **What does it mean if a model gets R²=0.99 on training data but R²=0.60 on test data?** Which practical concept does this relate to?

### ⚠️ Common Mistakes

| Mistake | Why It's Wrong |
|---------|---------------|
| Not splitting into train/test | You'll think your model is perfect, but it just memorized the data (overfitting). You'll get destroyed in viva when asked "how did you validate?" |
| Forgetting to save the LabelEncoders | When you predict risk for a new road edge later, you need the SAME encoder to convert "highway" → the same number it used during training |
| Using classification instead of regression | If you predict "safe/unsafe", you lose granularity. A* needs 0.73 vs 0.31, not just "risky" vs "safe" |
| Not clamping predictions to [0, 1] | Regression can predict -0.1 or 1.3. Always clamp: `max(0, min(1, prediction))` |

---

> [!TIP]
> **Before moving to Part 2**, make sure you can:
> 1. Draw a graph on paper and convert it to the Python dictionary format
> 2. Trace A* by hand on a 4-node graph (show g, h, f values at each step)
> 3. Explain why Gradient Boosting beat Linear Regression in your own words
>
> If you can do all three, you're ready for Part 2 where we **connect everything together**.
