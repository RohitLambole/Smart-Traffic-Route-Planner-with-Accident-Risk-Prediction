# Smart Traffic Route Planner with Accident Risk Prediction

> An intelligent system that predicts road-level accident risk using Machine Learning and uses that prediction inside A* Search to find safe routes, controlled by a goal-based AI Agent — now with a full Streamlit Web UI.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Real-World Applications](#real-world-applications)
- [How Technologies Connect to This Project](#how-technologies-connect-to-this-project)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [How to Present This Project to Your Teacher](#how-to-present-this-project-to-your-teacher)

---

## Project Overview

Traditional GPS systems find the **shortest** route. They ignore whether that route is **safe**.

This project solves that problem by:
1. **Predicting** the accident risk of every road using a trained Machine Learning model (Gradient Boosting, R²=0.9177)
2. **Finding** the safest optimal path using the A* Search Algorithm with a risk-aware cost function
3. **Deciding** intelligently whether to proceed or reroute using an AI Agent with memory
4. **Visualizing** everything through a professional dark-themed Streamlit Web UI

---

## Technologies Used

### 1. Python
**What it is:** A high-level, general-purpose programming language widely used in AI, ML, and data science.

**Real-World Use:**
- Google uses Python for internal AI/ML pipelines
- NASA uses Python for data analysis in space missions
- Instagram's backend is built in Python

**In This Project:**
- Core language for all modules — data generation, model training, A* algorithm, agent logic, visualization, and web UI

---

### 2. NumPy
**What it is:** A Python library for fast numerical computation using arrays and matrices.

**Real-World Use:**
- Used in financial modeling for portfolio risk calculations
- Used in image processing (pixel arrays)
- NASA uses it for astronomical data analysis

**In This Project:**
- Generates synthetic road risk data using `np.random.seed(42)` for reproducibility
- Clips prediction values to the valid range [0.0, 1.0]

---

### 3. Pandas
**What it is:** A Python library for data manipulation and analysis using DataFrames.

**Real-World Use:**
- Used by banks to analyze transaction datasets
- Used in healthcare to manage patient records
- Used in e-commerce to process order histories

**In This Project:**
- Loads and structures the 800-row road risk dataset
- Passes named DataFrames to the ML model (prevents sklearn feature name warnings)
- Exports the dataset to CSV for reuse

---

### 4. Scikit-learn
**What it is:** A Python ML library providing ready-to-use implementations of regression, classification, and preprocessing tools.

**Real-World Use:**
- Netflix uses similar ML pipelines for recommendation engines
- Banks use it for credit risk scoring
- Hospitals use it for disease likelihood prediction

**In This Project:**
- Trains 4 regression models to predict accident risk scores
- Provides `LabelEncoder` to convert text features (rain, highway) to numbers
- Evaluates models using MSE, RMSE, and R² metrics

---

### 5. Gradient Boosting Regressor (Ensemble Learning)
**What it is:** An ensemble ML technique that builds multiple decision trees sequentially, where each tree corrects the errors of the previous one.

**Real-World Use:**
- Used by Uber to predict trip demand surges
- Used by Amazon for product return probability prediction
- Used in weather forecasting for precipitation intensity prediction

**In This Project:**
- Achieved R² = 0.9177 — best among all 4 trained models
- Captures complex feature interactions like rain + highway + night = very high risk
- Its predictions are used as the safety input to the A* algorithm

---

### 6. A* Search Algorithm
**What it is:** A graph-based pathfinding algorithm that finds the optimal path from source to destination using f(n) = g(n) + h(n).

**Real-World Use:**
- Google Maps uses A* variants for route planning
- Video game AI (NPC pathfinding)
- Amazon Robotics warehouse path planning
- Drone delivery path optimization

**In This Project:**
- Finds the optimal path through a 15-node city road graph
- Uses Euclidean distance as the admissible heuristic
- Cost function: `weight = 0.6 × distance + 0.4 × (risk × 10)`

---

### 7. Goal-Based AI Agent
**What it is:** An intelligent software agent with a defined goal, perceiving the environment through sensors and acting through actuators.

**Real-World Use:**
- Self-driving cars (Tesla Autopilot)
- Warehouse robots (Amazon Kiva)
- Smart thermostats (Nest)
- Customer service chatbots

**In This Project:**
- **Goal:** Reach destination with all edge risks below 0.7
- **Sensors:** ML model predictions, weather, time, traffic density
- **Actuators:** Route selection, edge removal, rerouting
- **Memory:** Stores all past routing decisions in `self.memory[]`

Agent Decision Flow:
```
Plan Route via A*
      |
Scan each edge for risk > 0.7
      |
   Danger?
   /      \
  NO       YES
  |          |
PROCEED   Remove edge → Re-run A*
               |
          Found alt?
          /        \
        YES          NO
      REROUTE     PROCEED WITH CAUTION
```

---

### 8. Matplotlib + NetworkX
**What it is:** Libraries for creating visualizations and graph structures.

**Real-World Use:**
- Scientists plot experimental data with Matplotlib
- Telecom companies design network topology with NetworkX

**In This Project:**
- Draws the 15-node city road graph
- Colors edges: Red (risk > 0.7), Orange (0.4–0.7), Green (< 0.4)
- Highlights shortest path (blue dashed) vs agent's safe path (green solid)
- Saves `route_map.png` and 3 analysis graphs

---

### 9. Streamlit
**What it is:** A Python framework for building interactive web applications with zero frontend code.

**Real-World Use:**
- Data scientists use it to deploy ML models as web apps
- Used for internal dashboards in companies like Airbnb and Spotify

**In This Project:**
- Provides the full Web UI for the project (`app.py`)
- Interactive sidebar for source, destination, weather, time, traffic inputs
- Displays live route map, edge risk scan, path comparison, and ML graphs
- Dark-themed professional interface accessible at `http://localhost:8501`

---

### 10. Joblib
**What it is:** A Python library for saving and loading large Python objects, particularly ML models.

**Real-World Use:**
- Used in production ML systems to save trained models once and deploy many times

**In This Project:**
- Saves trained Gradient Boosting model to `risk_model.pkl`
- Saves encoders to `le_road.pkl` and `le_weather.pkl`
- Loads them at runtime — no retraining needed on every run

---

## Real-World Applications

| Application | Industry | How This Project Relates |
|-------------|----------|--------------------------|
| Google Maps / Waze Route Safety | Navigation | Risk-aware routing |
| Tesla Autopilot Path Planning | Automotive | Agent-based decision with sensor inputs |
| Amazon Delivery Route Optimization | Logistics | A* for shortest safe path |
| Smart City Traffic Management | Government | Real-time risk assessment per road |
| Insurance Premium Calculation | Finance | Road risk scoring per segment |
| Emergency Vehicle Routing | Healthcare | Fastest AND safest path to hospital |

---

## How Technologies Connect to This Project

```
+------------------+       +-------------------+       +------------------+
|   DATA LAYER     |       |    ML LAYER        |       |   AI LAYER       |
|                  |       |                    |       |                  |
|  NumPy           | ----> |  Scikit-learn      | ----> |  A* Algorithm    |
|  Pandas          |       |  Gradient Boosting |       |  AI Agent        |
|  CSV Dataset     |       |  (R² = 0.9177)     |       |  Memory + Reroute|
|  (800 samples)   |       |  Predicts risk 0-1 |       |                  |
+------------------+       +-------------------+       +------------------+
                                                                |
                                                                v
                                                  +------------------+
                                                  |  OUTPUT LAYER    |
                                                  |                  |
                                                  |  Streamlit UI    |
                                                  |  Matplotlib      |
                                                  |  NetworkX        |
                                                  |  route_map.png   |
                                                  |  3 graphs        |
                                                  +------------------+
```

---

## Project Structure

```
paai_mini/
|-- data/
|   |-- generate_data.py        # Creates 800-sample synthetic dataset (seed=42)
|   |-- road_risk_data.csv      # Generated training data (800 rows, 9 columns)
|-- models/
|   |-- risk_model.pkl          # Trained Gradient Boosting model (saved by model.py)
|   |-- le_road.pkl             # Road type label encoder
|   |-- le_weather.pkl          # Weather label encoder
|-- src/
|   |-- graph.py                # 15-node city graph with edge metadata
|   |-- a_star.py               # A* pathfinding with pluggable cost function
|   |-- model.py                # Train and compare 4 regression models
|   |-- risk_routing.py         # Bridge: ML model -> A* cost function
|   |-- agent.py                # Goal-based agent with memory and rerouting
|   |-- visualize.py            # Route map visualization (saves route_map.png)
|   |-- generate_graphs.py      # Generates 3 ML analysis graphs
|-- app.py                      # Streamlit Web UI (run: python -m streamlit run app.py)
|-- main.py                     # CLI entry point (interactive + demo modes)
|-- route_map.png               # Generated route visualization
|-- graph_model_comparison.png  # Bar chart: 4 models vs MSE/RMSE/R²
|-- graph_risk_distribution.png # Histogram + boxplot of risk by road type
|-- graph_feature_importance.png# Feature importance from Gradient Boosting
|-- requirements.txt            # Python dependencies (including streamlit)
|-- .gitignore                  # Excludes .pkl, __pycache__, venv
|-- README.md                   # This file
```

---

## How to Run

### First Time Setup (Run Once)

```
# Step 1: Install all dependencies
pip install -r requirements.txt

# Step 2: Generate training data
python data/generate_data.py

# Step 3: Train the ML models
python src/model.py

# Step 4: Generate result graphs
python src/generate_graphs.py
```

### Option A — Streamlit Web UI (Recommended for Demo)

```
python -m streamlit run app.py
```
Open browser at: **http://localhost:8501**

### Option B — Command Line (Interactive)

```
python main.py
```

### Option C — Command Line (Auto Demo — 3 scenarios)

```
python main.py --demo
```

### View Route Map

```
start route_map.png
```

---

## How to Present This Project to Your Teacher

Follow this flow for a 15-20 minute presentation.

---

### STEP 1 — Start with the Problem (1 min)

Say:
> "My mini project is a Smart Traffic Route Planner with Accident Risk Prediction. The problem is that Google Maps finds the shortest route — but the shortest route is not always the safest. It ignores weather, road type, and time of night. My system fixes that."

---

### STEP 2 — Open the Web UI (30 sec)

Run in terminal:
```
python -m streamlit run app.py
```
Open browser at `http://localhost:8501`

Say:
> "This is the web interface I built using Streamlit. On the left sidebar you can set the source, destination, weather, hour, and traffic."

---

### STEP 3 — Explain the Architecture (2 min)

Point to the README Technology Connection Diagram and say:
> "The system has 4 layers. The Data Layer creates 800 road samples. The ML Layer uses Gradient Boosting to predict risk scores between 0 and 1. The AI Layer uses A* to find the best path using those risk scores. The Output Layer shows the result through this web UI and the colored route map."

---

### STEP 4 — Live Demo — Dangerous Scenario (5 min)

In the Web UI sidebar, set:
```
Source:      B
Destination: N
Hour:        1
Weather:     Fog
Traffic:     0.5
```
Click **"Find Safe Route"**

While it runs, say:
> "I chose B to N, foggy weather at 1 AM with moderate traffic — a dangerous scenario. Watch what the agent decides."

Explain the output:

| What appears | What to say |
|-------------|-------------|
| Orange/Red REROUTE banner | "The agent detected danger and rerouted." |
| Edge risk scan: B→E risk=0.933 DANGER | "Highway in fog at 1 AM — 93% accident risk." |
| Path comparison table | "Normal GPS would take this shorter path. Our system avoids the dangerous highway." |
| Route Map | "Red roads are dangerous. Green line is the agent's safe route. Blue dashed is what normal GPS would give." |
| ML Graphs below | "These show why Gradient Boosting was selected — R² jumped from 0.55 to 0.92." |

---

### STEP 5 — Show Model Results (2 min)

Scroll down in the Web UI to the ML Analysis section. Point to the model comparison graph:

> "I trained 4 models. Linear Regression got R²=0.55 because accident risk is non-linear. Gradient Boosting got R²=0.9177 because it captures combinations — like rain plus highway plus midnight together being far more dangerous. The feature importance graph shows which factors matter most — speed limit and weather have the highest impact."

---

### STEP 6 — Show CLI Demo (optional, 2 min)

Open VS Code terminal:
```
python main.py --demo
```
Say:
> "This is the command-line version that runs 3 pre-set scenarios automatically — showing all 3 possible agent decisions: PROCEED, REROUTE, and PROCEED WITH CAUTION."

---

### STEP 7 — Close with 3 Key Points (1 min)

Say:
> "To summarize — this project uses Machine Learning to predict road risk, A* to find the optimal path, and an AI Agent to make the final routing decision. It connects Practicals 1, 2, 3, and 5 into one complete working system with a live web interface, result graphs, and a GitHub repository."

---

### Quick Answer Guide for Teacher Questions

| Question | Answer |
|----------|--------|
| Why A* over Dijkstra? | A* uses a heuristic — explores fewer nodes, faster and smarter |
| Why Gradient Boosting? | Captures compound interactions like rain+highway+night — Linear Regression cannot |
| Where is the ML part? | src/model.py — trains and saves the regression model |
| Where is the AI Agent? | src/agent.py — goal-based with memory |
| What does alpha/beta do? | Controls balance between distance and safety in the cost function |
| Is the data real? | Synthetic with real-world logical rules — designed to demonstrate the full pipeline |
| Where is memory stored? | In self.memory list inside TrafficAgent, active during the session |
| What is the Web UI built with? | Streamlit — a Python framework for building ML web apps |
| Why commit PNG graphs to GitHub? | So teacher can see results without running code — GitHub renders them automatically |

---

*Built as a 3rd Year Computer Engineering Mini Project combining AI Practicals 1, 2, 3, and 5.*
