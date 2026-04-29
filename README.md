# Smart Traffic Route Planner with Accident Risk Prediction

> An intelligent system that predicts road-level accident risk using Machine Learning and uses that prediction inside A* Search to find safe routes, controlled by a goal-based AI Agent.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Real-World Applications](#real-world-applications)
- [How Technologies Connect to This Project](#how-technologies-connect-to-this-project)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)

---

## Project Overview

Traditional GPS systems find the **shortest** route. They ignore whether that route is **safe**.

This project solves that problem by:
1. **Predicting** the accident risk of every road using a trained Machine Learning model
2. **Finding** the safest optimal path using the A* Search Algorithm
3. **Deciding** intelligently whether to proceed or reroute using an AI Agent

---

## Technologies Used

### 1. Python
**What it is:** A high-level, general-purpose programming language widely used in AI, ML, and data science.

**Real-World Use:**
- Google uses Python for internal AI/ML pipelines
- NASA uses Python for data analysis in space missions
- Instagram's backend is built in Python

**In This Project:**
- Core language for all modules — data generation, model training, A* algorithm, agent logic, and visualization

---

### 2. NumPy
**What it is:** A Python library for fast numerical computation using arrays and matrices.

**Real-World Use:**
- Used in financial modeling for portfolio risk calculations
- Used in image processing (pixel arrays)
- NASA uses it for astronomical data analysis

**In This Project:**
- Generates synthetic road risk data
- Performs element-wise operations to compute accident risk scores
- Clips prediction values to the valid range [0.0, 1.0]

---

### 3. Pandas
**What it is:** A Python library for data manipulation and analysis using DataFrames (like Excel tables in code).

**Real-World Use:**
- Used by banks to analyze transaction datasets
- Used in healthcare to manage patient records
- Used in e-commerce to process order histories

**In This Project:**
- Loads and structures the 800-row road risk dataset
- Handles feature columns (road_type, weather, hour_of_day, etc.)
- Exports the dataset to CSV for reuse

---

### 4. Scikit-learn
**What it is:** A Python ML library providing ready-to-use implementations of regression, classification, clustering, and preprocessing tools.

**Real-World Use:**
- Netflix uses similar ML pipelines for recommendation engines
- Banks use it for credit risk scoring
- Hospitals use it for disease likelihood prediction

**In This Project:**
- Trains 4 regression models to predict accident risk scores
- Provides `LabelEncoder` to convert text features (rain, highway) to numbers
- Evaluates models using MSE, RMSE, and R² metrics
- `train_test_split` ensures fair model evaluation

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
**What it is:** A graph-based pathfinding algorithm that finds the optimal path from a source to a destination using a cost function f(n) = g(n) + h(n), where g is actual cost and h is a heuristic estimate.

**Real-World Use:**
- Google Maps uses A* variants for route planning
- Video game AI (NPCs finding paths in open worlds)
- Robotics — robot arm path planning in warehouses (Amazon Robotics)
- Drone delivery path optimization

**In This Project:**
- Finds the optimal path through a 15-node city road graph
- Uses Euclidean distance as the admissible heuristic
- Cost function is modified to include both distance AND predicted accident risk:

```
weight = 0.6 × distance + 0.4 × (risk_score × 10)
```

This means safer roads are preferred even if they are slightly longer.

---

### 7. Goal-Based AI Agent
**What it is:** An intelligent software agent that has a defined goal, perceives its environment through sensors, and takes actions using actuators to achieve the goal.

**Real-World Use:**
- Self-driving cars (Tesla Autopilot) — goal: reach destination safely
- Warehouse robots (Amazon Kiva) — goal: pick and place items optimally
- Customer service chatbots — goal: resolve user query
- Smart thermostats (Nest) — goal: maintain comfortable temperature efficiently

**In This Project:**
- **Goal:** Reach destination with no road segment having risk > 0.7
- **Sensors:** ML model predictions, current weather, time, traffic density
- **Actuators:** Plan route, flag danger, remove risky edges, trigger reroute
- **Memory:** Stores all past routing decisions within a session

Agent Decision Flow:
```
Plan Route via A*
      |
Scan each edge for risk
      |
   Risk > 0.7?
   /         \
 NO           YES
  |             |
PROCEED     Remove edge
             Re-run A*
               |
          Found alt?
          /        \
        YES          NO
      REROUTE     PROCEED WITH
                   CAUTION
```

---

### 8. Matplotlib
**What it is:** A Python library for creating static, animated, and interactive visualizations.

**Real-World Use:**
- Scientists use it to plot experimental data
- Financial analysts use it for stock trend visualization
- Weather agencies use it for climate data charts

**In This Project:**
- Draws the 15-node city road graph
- Colors edges Red (high risk), Orange (medium risk), Green (safe)
- Highlights the shortest path (blue dashed) vs the agent's chosen safe path (green solid)
- Saves the route map as `route_map.png`

---

### 9. NetworkX
**What it is:** A Python library for creating, manipulating, and studying complex networks and graphs.

**Real-World Use:**
- Used in social network analysis (finding influential users)
- Used in biology for protein interaction networks
- Used in telecommunications for network topology design

**In This Project:**
- Builds the city road graph structure for visualization
- Provides the graph object that matplotlib draws on

---

### 10. Joblib
**What it is:** A Python library for efficient serialization (saving/loading) of large Python objects, particularly ML models.

**Real-World Use:**
- Used in production ML systems to save trained models once and deploy them many times
- Used in scientific computing to cache expensive computations

**In This Project:**
- Saves the trained Gradient Boosting model to `risk_model.pkl`
- Saves the LabelEncoders to `le_road.pkl` and `le_weather.pkl`
- Loads them back at runtime so the model does not need to be retrained on every run

---

## Real-World Applications

| Application | Industry | How This Project Relates |
|-------------|----------|--------------------------|
| Google Maps / Waze Route Safety | Navigation | Same concept — risk-aware routing |
| Tesla Autopilot Path Planning | Automotive | Agent-based decision with sensor inputs |
| Amazon Delivery Route Optimization | Logistics | A* for shortest safe delivery path |
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
|  CSV Dataset     |       |  (R2 = 0.9177)     |       |  Decision Logic  |
|  (800 samples)   |       |  Predicts risk 0-1 |       |  Rerouting       |
+------------------+       +-------------------+       +------------------+
                                                                |
                                                                v
                                                  +------------------+
                                                  |  OUTPUT LAYER    |
                                                  |                  |
                                                  |  Matplotlib      |
                                                  |  NetworkX        |
                                                  |  route_map.png   |
                                                  +------------------+
```

Every technology feeds into the next:
- **Pandas + NumPy** create the training data
- **Scikit-learn** trains the model on that data
- **Joblib** saves the trained model
- **The saved model** predicts risk scores at runtime
- **A*** uses those risk scores in its cost function
- **The Agent** uses A* to plan routes and make decisions
- **Matplotlib + NetworkX** visualize the final result

---

## Project Structure

```
paai_mini/
|-- data/
|   |-- generate_data.py      # Creates 800-sample synthetic dataset
|   |-- road_risk_data.csv    # Generated training data
|-- models/
|   |-- risk_model.pkl        # Trained Gradient Boosting model
|   |-- le_road.pkl           # Road type encoder
|   |-- le_weather.pkl        # Weather encoder
|-- src/
|   |-- graph.py              # 15-node city graph with edge metadata
|   |-- a_star.py             # A* pathfinding algorithm
|   |-- model.py              # Train and compare 4 regression models
|   |-- risk_routing.py       # Bridge: ML model -> A* cost function
|   |-- agent.py              # Goal-based agent with memory
|   |-- visualize.py          # Route map visualization
|-- main.py                   # Entry point (interactive + demo modes)
|-- route_map.png             # Generated route map
|-- requirements.txt          # Python dependencies
|-- README.md                 # This file
```

---

## How to Run

Open VS Code terminal in the project folder and run:

```
# Step 1: Install dependencies (once)
pip install -r requirements.txt

# Step 2: Generate training data (once)
python data/generate_data.py

# Step 3: Train the ML models (once)
python src/model.py

# Step 4: Run the simulation (interactive)
python main.py

# OR run the auto demo
python main.py --demo

# Step 5: View the route map
start route_map.png
```

---

## How to Present This Project to Your Teacher

Follow this flow for a 10-12 minute presentation.

---

### Step 1: Start with the Problem (1 minute)

Say:
> "My mini project is called Smart Traffic Route Planner with Accident Risk Prediction. The problem is simple — Google Maps finds the shortest route, but the shortest route is not always the safest. It ignores weather, road type, and time of night. My system fixes that."

No screen needed at this point.

---

### Step 2: Show the README (30 seconds)

Open `README.md` in VS Code.

Say:
> "This is my project overview. It combines 4 AI practicals — Agents, A* Search, Regression, and Ensemble Learning — into one complete working system. Each technology has a real-world connection."

Point to the Technology Connection Diagram section.

---

### Step 3: Show the Project Structure (30 seconds)

Point to the folder tree visible in VS Code's left panel.

Say:
> "The project is structured professionally. Data is separate, models are saved, all logic is in individual source files. main.py is the single entry point."

---

### Step 4: Run the Live Simulation (5-6 minutes)

Open the VS Code terminal and type:

```
python main.py
```

When prompted, enter these values and explain each:

```
Source node:      B
Destination node: N
Hour:             1
Weather:          fog
Traffic:          0.5
```

Say while typing:
> "I am choosing B to N, foggy weather at 1 AM with moderate traffic — a dangerous real-world scenario. Watch what the agent decides."

Explain the output line by line as it appears:

| What appears on screen | What to say |
|------------------------|-------------|
| `[SHORTEST] B -> D -> E -> I -> J -> N` | "This is what a normal GPS gives — shortest by distance only." |
| `[SAFE] B -> E -> I -> J -> N` | "This is my system's initial risk-aware path using A*." |
| `B -> E: risk=0.933 !! DANGER` | "The ML model predicted 93% accident risk on this highway — extremely dangerous at 1 AM in fog." |
| `[REROUTE] Attempting reroute` | "The agent detected danger and removed that road from the graph." |
| `FINAL DECISION: REROUTE` | "It found a safer alternative automatically. This is the agent working." |
| `[MEMORY] Agent Memory` | "The agent also stores every decision it makes — this is the memory feature from Practical 1." |

---

### Step 5: Open the Route Map (1 minute)

Type in terminal:

```
start route_map.png
```

Say:
> "This is the visual output. Red edges are dangerous roads, orange is moderate, green is safe. The bright green line is the route my agent chose — it avoided the red highway. The blue dashed line is what a normal GPS would have taken."

---

### Step 6: Mention Model Training Results (1 minute)

Say:
> "During training, I compared 4 models — Linear Regression, Polynomial, Random Forest, and Gradient Boosting. Gradient Boosting gave R squared = 0.9177, meaning it explains 92% of the variance in accident risk. That is why it was selected automatically as the best model."

If asked, open `src/model.py` and point to the comparison table in the output.

---

### Step 7: Close with 3 Key Points (30 seconds)

Say:
> "To summarize — this project uses Machine Learning to predict road risk, A* to find the optimal path, and an AI Agent to make the final routing decision. It connects Practicals 1, 2, 3, and 5 into one complete working system with a live demo and visual output."

---

### Quick Answer Guide for Teacher Questions

| Question | Answer |
|----------|--------|
| Why A* over Dijkstra? | A* uses a heuristic and explores fewer nodes — faster and smarter |
| Why Gradient Boosting? | It captures compound interactions like rain + highway + night that Linear Regression cannot |
| Where is the ML part? | src/model.py — trains and saves the regression model |
| Where is the AI Agent? | src/agent.py — goal-based with memory |
| What does alpha/beta do? | Controls the balance between distance and safety in the cost function |
| Is the data real? | Synthetic with real-world logical rules — designed to demonstrate the full pipeline |
| Where is memory stored? | In self.memory list inside the TrafficAgent class, active during the session |

---

*Built as a 3rd Year Computer Engineering Mini Project combining AI Practicals 1, 2, 3, and 5.*
