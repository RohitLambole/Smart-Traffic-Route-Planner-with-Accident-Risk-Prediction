"""
=============================================================
  MAIN ENTRY POINT — Smart Traffic Route Planner
=============================================================
  This ties all modules together:
    1. Graph (Module 1)
    2. A* Algorithm (Module 2)
    3. Risk Prediction Model (Module 3)
    4. Risk-Aware Routing (Module 4)
    5. Agent Decision System (Module 5)
    6. Visualization (Module 6)

  Run modes:
    python main.py          → Interactive mode (you choose nodes)
    python main.py --demo   → Auto demo with 3 preset scenarios
=============================================================
"""

import sys
import os

# Add src/ to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from graph import city_graph, node_positions
from risk_routing import load_model, set_conditions, current_conditions
from agent import TrafficAgent
from visualize import visualize_routes


ALL_NODES = sorted(city_graph.keys())


def print_header():
    print("\n" + "=" * 58)
    print("   Smart Traffic Route Planner")
    print("   with Accident Risk Prediction")
    print("=" * 58)


def print_conditions():
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    c = current_conditions
    print(f"\n[Current Conditions]")
    print(f"   Time:    {c['hour_of_day']}:00")
    print(f"   Day:     {days[c['day_of_week']]}")
    print(f"   Weather: {c['weather']}")
    print(f"   Traffic: {c['traffic_density'] * 100:.0f}%")


def print_result(result):
    print(f"\n{'=' * 58}")
    print(f"  FINAL DECISION: {result['action'].upper()}")
    print(f"  Chosen Path:    {' -> '.join(result['path'])}")
    print(f"  Path Cost:      {result['cost']:.2f}")
    if result.get('shortest_path'):
        print(f"  Shortest Path:  {' -> '.join(result['shortest_path'])} (cost: {result['shortest_cost']:.2f})")
    print(f"  {result['message']}")
    print(f"{'=' * 58}")


def print_city_map():
    """Show available nodes and their connections."""
    print(f"\n[City Map] — {len(ALL_NODES)} intersections")
    print(f"   Available nodes: {', '.join(ALL_NODES)}")
    print(f"\n   Connections:")
    for node in ALL_NODES:
        neighbors = ', '.join([f"{n}({d}km)" for n, d in city_graph[node]])
        print(f"   {node} -> {neighbors}")


def get_user_input():
    """Get source, destination, and conditions from user."""

    print_city_map()

    # --- Get source node ---
    while True:
        source = input(f"\n>> Enter SOURCE node ({', '.join(ALL_NODES)}): ").strip().upper()
        if source in ALL_NODES:
            break
        print(f"   Invalid! Choose from: {', '.join(ALL_NODES)}")

    # --- Get destination node ---
    while True:
        dest = input(f">> Enter DESTINATION node ({', '.join(ALL_NODES)}): ").strip().upper()
        if dest in ALL_NODES and dest != source:
            break
        if dest == source:
            print("   Destination must be different from source!")
        else:
            print(f"   Invalid! Choose from: {', '.join(ALL_NODES)}")

    # --- Get time ---
    while True:
        hour_str = input(">> Enter hour of day (0-23) [default: 14]: ").strip()
        if hour_str == '':
            hour = 14
            break
        try:
            hour = int(hour_str)
            if 0 <= hour <= 23:
                break
            print("   Must be 0-23!")
        except ValueError:
            print("   Enter a number!")

    # --- Get weather ---
    while True:
        weather = input(">> Enter weather (clear/rain/fog) [default: clear]: ").strip().lower()
        if weather == '':
            weather = 'clear'
            break
        if weather in ['clear', 'rain', 'fog']:
            break
        print("   Choose: clear, rain, or fog")

    # --- Get traffic ---
    while True:
        traffic_str = input(">> Enter traffic density 0.1-1.0 (e.g. 0.5) [default: 0.5]: ").strip()
        if traffic_str == '':
            traffic = 0.5
            break
        try:
            traffic = float(traffic_str)
            if 0.1 <= traffic <= 1.0:
                break
            print("   Must be between 0.1 and 1.0!")
        except ValueError:
            print("   Enter a decimal number!")

    return source, dest, hour, weather, traffic


def run_interactive():
    """Interactive mode — user chooses nodes and conditions."""
    print_header()

    if not load_model():
        print("\n[ERROR] Cannot start without trained model.")
        print("   Run these commands first:")
        print("   1. python data/generate_data.py")
        print("   2. python src/model.py")
        return

    agent = TrafficAgent(city_graph, node_positions, risk_threshold=0.7)

    while True:
        # Get user input
        source, dest, hour, weather, traffic = get_user_input()

        # Set conditions
        set_conditions(hour=hour, day=3, weather=weather, traffic=traffic)
        print_conditions()

        # Plan route
        result = agent.plan_route(source, dest)
        print_result(result)

        # Visualize
        save_path = os.path.join(os.path.dirname(__file__), 'route_map.png')
        try:
            visualize_routes(city_graph, node_positions, result, save_path=save_path)
        except Exception as e:
            print(f"   Visualization note: {e}")

        # Show memory
        agent.show_memory()

        # Ask to continue
        print("\n" + "-" * 58)
        again = input(">> Try another route? (y/n): ").strip().lower()
        if again != 'y':
            print("\nThank you for using Smart Traffic Route Planner!")
            break


def run_demo():
    """Auto demo with 3 preset scenarios — no input needed."""
    print_header()
    print("\n   [DEMO MODE] — Running 3 preset scenarios")

    if not load_model():
        print("\n[ERROR] Cannot start without trained model.")
        print("   Run these commands first:")
        print("   1. python data/generate_data.py")
        print("   2. python src/model.py")
        return

    agent = TrafficAgent(city_graph, node_positions, risk_threshold=0.7)

    # ---- Scenario 1: Dangerous night + rain ----
    print("\n" + ">" * 20 + " SCENARIO 1: Rainy Night " + "<" * 20)
    set_conditions(hour=23, day=5, weather='rain', traffic=0.7)
    print_conditions()
    result1 = agent.plan_route('A', 'O')
    print_result(result1)

    # ---- Scenario 2: Safe daytime + clear ----
    print("\n" + ">" * 20 + " SCENARIO 2: Clear Morning " + "<" * 20)
    set_conditions(hour=10, day=2, weather='clear', traffic=0.3)
    print_conditions()
    result2 = agent.plan_route('A', 'O')
    print_result(result2)

    # ---- Scenario 3: Different route ----
    print("\n" + ">" * 20 + " SCENARIO 3: Foggy Midnight " + "<" * 20)
    set_conditions(hour=1, day=6, weather='fog', traffic=0.5)
    print_conditions()
    result3 = agent.plan_route('B', 'N')
    print_result(result3)

    # ---- Show agent memory ----
    agent.show_memory()

    # ---- Visualize ----
    print("\nGenerating visualization for Scenario 1...")
    save_path = os.path.join(os.path.dirname(__file__), 'route_map.png')
    try:
        visualize_routes(city_graph, node_positions, result1, save_path=save_path)
    except Exception as e:
        print(f"   Visualization note: {e}")
        print("   Map saved to route_map.png")


if __name__ == '__main__':
    if '--demo' in sys.argv:
        run_demo()
    else:
        run_interactive()
