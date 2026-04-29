# HOW TO RUN — Complete Step-by-Step Guide

This guide walks you through running the **Smart Traffic Route Planner** from scratch using the **VS Code integrated terminal**.

---

## Prerequisites

Make sure you have **Python 3.8+** installed. Open VS Code terminal (Ctrl + `) and type:

```
python --version
```

You should see something like `Python 3.14.0`. If you get "not recognized", install Python from https://python.org/downloads (check "Add Python to PATH" during install).

---

## Step 1: Open Project in VS Code

1. Open **VS Code**
2. Go to **File > Open Folder**
3. Select the folder: `C:\Users\Asus\Desktop\paai_mini`
4. Press **Ctrl + `** (backtick) to open the integrated terminal
5. The terminal should show: `PS C:\Users\Asus\Desktop\paai_mini>`

---

## Step 2: Install Dependencies (One Time Only)

In the VS Code terminal, type:

```
pip install -r requirements.txt
```

You should see packages being installed or "Requirement already satisfied" messages. Both are fine.

---

## Step 3: Generate Training Dataset

This creates 800 synthetic road samples with accident risk patterns.

```
python data/generate_data.py
```

**You should see:**
```
Generating synthetic road risk data...
[OK] Saved 800 samples to: ...\data\road_risk_data.csv

--- Sample Data ---
  road_type  hour_of_day  ...  accident_risk
residential            8  ...          0.316
       lane           16  ...          0.660
    highway           16  ...          0.323
...

--- Risk Statistics ---
   Mean:  0.468
   Min:   0.000
   Max:   1.000
```

**What was created:** `data/road_risk_data.csv` (800 rows of training data)

---

## Step 4: Train the ML Models

This trains 4 models, compares them, and saves the best one.

```
python src/model.py
```

**You should see:**
```
[DATA] Loaded 800 samples

============================================================
  Model                          MSE     RMSE       R2
============================================================
  Linear Regression           0.0136   0.1164   0.5508
  Polynomial (degree 2)       0.0061   0.0784   0.7964
  Random Forest               0.0046   0.0678   0.8478
  Gradient Boosting           0.0025   0.0498   0.9177
============================================================

>> Best Model: Gradient Boosting (R2 = 0.9177)

[OK] Saved to ...\models/:
   - risk_model.pkl
   - le_road.pkl
   - le_weather.pkl
```

**What was created:** 3 files in `models/` folder (trained model + encoders)

---

## Step 5: Run the Simulation (TWO MODES)

### Mode A: Interactive Mode (YOU choose nodes and conditions)

```
python main.py
```

**What happens:**
1. The system shows you the **city map** with all 15 nodes (A to O) and their connections
2. You **type the source node** (e.g., `A`)
3. You **type the destination node** (e.g., `O`)
4. You **choose conditions**: hour, weather, traffic density
5. The agent finds the route and makes a decision
6. A route map image is saved
7. It asks if you want to try another route

**Example interaction:**
```
==========================================================
   Smart Traffic Route Planner
   with Accident Risk Prediction
==========================================================

[City Map] -- 15 intersections
   Available nodes: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O

   Connections:
   A -> B(3.0km), C(4.5km)
   B -> A(3.0km), D(2.5km), E(5.0km)
   ...

>> Enter SOURCE node (A, B, C, ...): B
>> Enter DESTINATION node (A, B, C, ...): N
>> Enter hour of day (0-23) [default: 14]: 1
>> Enter weather (clear/rain/fog) [default: clear]: fog
>> Enter traffic density 0.1-1.0 (e.g. 0.5) [default: 0.5]: 0.5

[Current Conditions]
   Time:    1:00
   Day:     Thu
   Weather: fog
   Traffic: 50%

--------------------------------------------------
[AGENT] Planning route B -> N
   Threshold: risk > 0.7 = dangerous
--------------------------------------------------

[SHORTEST] Path (distance only):
   B -> D -> E -> I -> J -> N (cost: 13.00)

[SAFE] Risk-aware path:
   B -> E -> I -> J -> N (cost: 19.50)

[SCAN] Scanning route for dangers...
   B -> E: risk=0.933 [highway, 80km/h] !! DANGER
   E -> I: risk=0.689 [main_road, 60km/h] OK
   I -> J: risk=0.545 [residential, 30km/h] OK
   J -> N: risk=0.608 [main_road, 40km/h] OK

[WARNING] 1 segment(s) above threshold!

[REROUTE] Attempting reroute (removing 1 edge(s))...
   >> Alternative found: B -> D -> E -> I -> J -> N (cost: 20.11)

==========================================================
  FINAL DECISION: REROUTE
  Chosen Path:    B -> D -> E -> I -> J -> N
  Path Cost:      20.11
  Shortest Path:  B -> D -> E -> I -> J -> N (cost: 13.00)
  >> Rerouted to avoid 1 dangerous segment(s).
==========================================================

----------------------------------------------------------
>> Try another route? (y/n): y

(... you can try different nodes and conditions again ...)

>> Try another route? (y/n): n

Thank you for using Smart Traffic Route Planner!
```

### Mode B: Auto Demo (3 preset scenarios, no input needed)

```
python main.py --demo
```

This runs 3 pre-configured scenarios automatically:
- **Scenario 1:** Rainy night (23:00, rain, 70% traffic) — route A to O
- **Scenario 2:** Clear morning (10:00, clear, 30% traffic) — route A to O
- **Scenario 3:** Foggy midnight (01:00, fog, 50% traffic) — route B to N

No typing needed — it runs all three and shows results.

---

## Step 6: View the Route Map

After running either mode, a map image is saved. Open it:

**Option 1:** Just look in the VS Code file explorer (left sidebar) — click on `route_map.png`

**Option 2:** In terminal:
```
start route_map.png
```

**What the map shows:**
- **Green edges** = safe roads (risk < 0.4)
- **Orange edges** = moderate risk (0.4-0.7)
- **Red edges** = dangerous roads (risk > 0.7)
- **Bright green line** = agent's chosen route
- **Blue dashed line** = shortest distance path
- Numbers on edges = predicted risk scores

---

## Suggested Demo Combinations to Try

Try these in interactive mode to see different agent behaviors:

| Source | Dest | Hour | Weather | Traffic | What You'll See |
|:------:|:----:|:----:|---------|:-------:|-----------------|
| A | O | 14 | clear | 0.3 | Safe day — PROCEED |
| A | O | 23 | rain | 0.8 | Risky night — check if it reroutes |
| B | N | 1 | fog | 0.5 | Highway danger — REROUTE |
| A | L | 2 | rain | 0.9 | Through highways — likely REROUTE |
| G | L | 10 | clear | 0.2 | Cross-city safe route — PROCEED |
| H | O | 0 | fog | 0.7 | Late night fog — high risk edges |

---

## Quick Reference: All Commands

```
# STEP 1: Generate data (run once)
python data/generate_data.py

# STEP 2: Train model (run once)
python src/model.py

# STEP 3a: Interactive simulation (you choose nodes)
python main.py

# STEP 3b: Auto demo (3 preset scenarios)
python main.py --demo

# STEP 4: Open the route map
start route_map.png
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python not recognized` | Install Python, check "Add to PATH" |
| `ModuleNotFoundError: sklearn` | Run `pip install scikit-learn` |
| `Dataset not found` | Run `python data/generate_data.py` first |
| `Model files not found` | Run `python src/model.py` first |
| Warnings about "feature names" | Ignore — does not affect results |
| Map doesn't open automatically | Click `route_map.png` in VS Code explorer |

---

## Checklist — Is Everything Working?

- [ ] `python data/generate_data.py` prints "Saved 800 samples"
- [ ] `python src/model.py` shows Gradient Boosting as best model
- [ ] `python main.py` lets you type source/destination nodes
- [ ] Agent shows REROUTE when you use fog + highway at night
- [ ] `route_map.png` exists and shows colored graph
- [ ] `python main.py --demo` runs 3 scenarios without errors

**All boxes checked = project fully running!**
