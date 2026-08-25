"""
=============================================================
   MAIN ENTRY POINT — Smart Traffic Route Planner (OSM Edition)
=============================================================
   Extended version with OpenStreetMap real-world road data.
   
   This version can:
     - Use real OSM road networks (requires osmnx)
     - Fall back to synthetic graph if OSM unavailable
     - Demo with both graph sources
   
   Run modes:
     python main_osm.py                      → Interactive with OSM
     python main_osm.py --synthetic          → Use synthetic graph
     python main_osm.py --demo               → Demo with both
     python main_osm.py --city "San Francisco, California"
=============================================================
"""

import sys
import os
import argparse

# Add src/ to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from graph import city_graph, node_positions
from risk_routing import load_model, set_conditions, current_conditions
from agent import TrafficAgent
from visualize import visualize_routes

try:
    from osm_graph import load_osm_graph
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False
    print("[WARNING] osmnx not installed. Install with: pip install osmnx")


ALL_NODES_SYNTHETIC = sorted(city_graph.keys())


def print_header(title="Smart Traffic Route Planner"):
    print("\n" + "=" * 60)
    print(f"   {title}")
    print("   with Accident Risk Prediction")
    print("=" * 60)


def print_conditions():
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    c = current_conditions
    print(f"\n[Current Conditions]")
    print(f"   Time:    {c['hour_of_day']}:00")
    print(f"   Day:     {days[c['day_of_week']]}")
    print(f"   Weather: {c['weather']}")
    print(f"   Traffic: {c['traffic_density'] * 100:.0f}%")


def print_result(result):
    print(f"\n{'=' * 60}")
    print(f"  FINAL DECISION: {result['action'].upper()}")
    print(f"  Chosen Path:    {' -> '.join(str(n) for n in result['path'])}")
    print(f"  Path Cost:      {result['cost']:.2f}")
    if result.get('shortest_path'):
        sp_str = ' -> '.join(str(n) for n in result['shortest_path'])
        print(f"  Shortest Path:  {sp_str} (cost: {result['shortest_cost']:.2f})")
    print(f"  {result['message']}")
    print(f"{'=' * 60}")


def run_interactive_osm(location: str):
    """Interactive mode with OpenStreetMap data."""
    if not HAS_OSMNX:
        print("[ERROR] osmnx not installed. Run: pip install osmnx")
        return
    
    print_header("OSM Edition — Real Road Network")
    
    if not load_model():
        print("\n[ERROR] Cannot start without trained model.")
        print("   Run these commands first:")
        print("   1. python data/generate_data.py")
        print("   2. python src/model.py")
        return
    
    try:
        graph, positions = load_osm_graph(location, use_cache=True)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\nFalling back to synthetic graph...")
        graph, positions = city_graph, node_positions
    
    all_nodes = sorted(graph.keys())
    agent = TrafficAgent(graph, positions, risk_threshold=0.7)
    
    print(f"\n✓ Loaded graph with {len(all_nodes)} nodes")
    print(f"  Sample nodes: {all_nodes[:10]}")
    
    while True:
        # --- Get source node ---
        while True:
            source_input = input(f"\n>> Enter SOURCE node (or 'list' to see all): ").strip()
            if source_input.lower() == 'list':
                print(f"   All nodes: {all_nodes}")
                continue
            source = source_input
            if source in all_nodes:
                break
            print(f"   Invalid! Choose from the list or type 'list'")
        
        # --- Get destination node ---
        while True:
            dest_input = input(f">> Enter DESTINATION node: ").strip()
            dest = dest_input
            if dest in all_nodes and dest != source:
                break
            if dest == source:
                print("   Destination must be different from source!")
            else:
                print(f"   Invalid node!")
        
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
        
        # Set conditions and plan route
        set_conditions(hour=hour, day=3, weather=weather, traffic=traffic)
        print_conditions()
        
        result = agent.plan_route(source, dest)
        print_result(result)
        
        # Visualize
        save_path = os.path.join(os.path.dirname(__file__), 'route_map_osm.png')
        try:
            visualize_routes(graph, positions, result, save_path=save_path)
        except Exception as e:
            print(f"   Visualization note: {e}")
        
        # Show memory
        agent.show_memory()
        
        # Ask to continue
        print("\n" + "-" * 60)
        again = input(">> Try another route? (y/n): ").strip().lower()
        if again != 'y':
            print("\nThank you for using Smart Traffic Route Planner!")
            break


def run_interactive_synthetic():
    """Interactive mode with synthetic graph."""
    print_header("Synthetic Edition")
    
    if not load_model():
        print("\n[ERROR] Cannot start without trained model.")
        print("   Run these commands first:")
        print("   1. python data/generate_data.py")
        print("   2. python src/model.py")
        return
    
    agent = TrafficAgent(city_graph, node_positions, risk_threshold=0.7)
    
    while True:
        # Get user input
        print(f"\n[City Map] — {len(ALL_NODES_SYNTHETIC)} intersections")
        print(f"   Available nodes: {', '.join(ALL_NODES_SYNTHETIC)}")
        
        while True:
            source = input(f"\n>> Enter SOURCE node ({', '.join(ALL_NODES_SYNTHETIC)}): ").strip().upper()
            if source in ALL_NODES_SYNTHETIC:
                break
            print(f"   Invalid! Choose from: {', '.join(ALL_NODES_SYNTHETIC)}")
        
        while True:
            dest = input(f">> Enter DESTINATION node ({', '.join(ALL_NODES_SYNTHETIC)}): ").strip().upper()
            if dest in ALL_NODES_SYNTHETIC and dest != source:
                break
            if dest == source:
                print("   Destination must be different from source!")
            else:
                print(f"   Invalid! Choose from: {', '.join(ALL_NODES_SYNTHETIC)}")
        
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
        
        while True:
            weather = input(">> Enter weather (clear/rain/fog) [default: clear]: ").strip().lower()
            if weather == '':
                weather = 'clear'
                break
            if weather in ['clear', 'rain', 'fog']:
                break
            print("   Choose: clear, rain, or fog")
        
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
        print("\n" + "-" * 60)
        again = input(">> Try another route? (y/n): ").strip().lower()
        if again != 'y':
            print("\nThank you for using Smart Traffic Route Planner!")
            break


def run_demo():
    """Demo comparing both graph sources."""
    print_header("DEMO Mode — Both Synthetic & OSM")
    
    if not load_model():
        print("\n[ERROR] Cannot start without trained model.")
        return
    
    # === PART 1: Synthetic Graph ===
    print("\n" + ">" * 20 + " PART 1: Synthetic Graph Demo " + "<" * 20)
    agent_syn = TrafficAgent(city_graph, node_positions, risk_threshold=0.7)
    
    print("\n[Scenario 1] Rainy night (synthetic)")
    set_conditions(hour=23, day=5, weather='rain', traffic=0.7)
    print_conditions()
    result1 = agent_syn.plan_route('A', 'O')
    print_result(result1)
    
    print("\n[Scenario 2] Clear morning (synthetic)")
    set_conditions(hour=10, day=2, weather='clear', traffic=0.3)
    print_conditions()
    result2 = agent_syn.plan_route('A', 'O')
    print_result(result2)
    
    # === PART 2: OSM Graph (if available) ===
    if HAS_OSMNX:
        print("\n" + ">" * 20 + " PART 2: OpenStreetMap Demo " + "<" * 20)
        try:
            location = "San Francisco, California"
            print(f"\nAttempting to load OSM graph for: {location}")
            graph_osm, positions_osm = load_osm_graph(location, use_cache=True)
            
            agent_osm = TrafficAgent(graph_osm, positions_osm, risk_threshold=0.7)
            
            # Get two random accessible nodes
            all_osm_nodes = sorted(graph_osm.keys())
            source_osm = all_osm_nodes[0]
            dest_osm = all_osm_nodes[min(len(all_osm_nodes) - 1, 20)]
            
            print(f"\n[Scenario 3] Foggy conditions (OSM: {location})")
            print(f"  Route: {source_osm} → {dest_osm}")
            set_conditions(hour=1, day=6, weather='fog', traffic=0.5)
            print_conditions()
            result3 = agent_osm.plan_route(source_osm, dest_osm)
            print_result(result3)
            
            print("\n✓ OSM integration working!")
            
        except Exception as e:
            print(f"\n✗ OSM demo failed: {e}")
            print("  (This is expected if osmnx isn't installed or network is unavailable)")
    else:
        print("\n[SKIPPED] OSM demo (osmnx not installed)")
        print("  Install with: pip install osmnx")
    
    print("\n" + "=" * 60)
    print("[DEMO COMPLETE]")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Smart Traffic Route Planner — Synthetic or OSM Edition'
    )
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic graph')
    parser.add_argument('--osm', action='store_true', help='Use OpenStreetMap')
    parser.add_argument('--city', type=str, default='San Francisco, California',
                        help='City for OSM (format: "City, State/Country")')
    parser.add_argument('--demo', action='store_true', help='Run demo mode')
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.synthetic:
        run_interactive_synthetic()
    elif args.osm or (not args.synthetic and HAS_OSMNX):
        # Default to OSM if available
        run_interactive_osm(args.city)
    else:
        # Fall back to synthetic if osmnx not available
        print("[INFO] osmnx not installed. Using synthetic graph.")
        print("       To use OpenStreetMap, run: pip install osmnx")
        run_interactive_synthetic()
