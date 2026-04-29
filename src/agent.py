"""
=============================================================
  MODULE 5: Traffic Agent — The Decision Maker
=============================================================
  The agent sits on top of A* and the risk model.
  It doesn't find paths or predict risk — it DECIDES:
    - Is the route safe enough?
    - Should I reroute?
    - What did I decide before? (memory)

  Agent types demonstrated (from Practical 1):
    - Reactive: checks current risk → responds
    - Goal-based: aims for destination with safety constraint
    - Memory-based: stores past decisions
    - Environment-aware: reads weather, time, traffic
=============================================================
"""

import copy
from a_star import a_star
from risk_routing import predict_edge_risk, create_weight_fn, current_conditions
from graph import get_edge_metadata


class TrafficAgent:
    """
    Intelligent traffic routing agent.

    The agent acts as the decision layer:
    1. Gets a route from A* (with risk-aware weights)
    2. Scans the route for dangerously risky segments
    3. Decides: proceed / reroute / proceed with caution
    4. Remembers past decisions (memory)
    """

    def __init__(self, graph, positions, risk_threshold=0.7):
        self.graph = graph
        self.positions = positions
        self.risk_threshold = risk_threshold
        self.memory = []  # Past decisions — makes this a memory-based agent
        self.weight_fn = create_weight_fn(get_edge_metadata)

    def plan_route(self, source, destination):
        """
        Main decision function.
        Called by the user — agent handles everything internally.
        """
        print(f"\n{'-' * 50}")
        print(f"[AGENT] Planning route {source} -> {destination}")
        print(f"   Threshold: risk > {self.risk_threshold} = dangerous")
        print(f"{'-' * 50}")

        # Step 1: Find shortest path (distance only) for comparison
        shortest_path, shortest_cost = a_star(
            self.graph, self.positions, source, destination
        )
        print(f"\n[SHORTEST] Path (distance only):")
        if shortest_path:
            print(f"   {' -> '.join(shortest_path)} (cost: {shortest_cost:.2f})")

        # Step 2: Find risk-aware path
        safe_path, safe_cost = a_star(
            self.graph, self.positions, source, destination,
            weight_fn=self.weight_fn
        )

        if safe_path is None:
            print("[ERROR] No path found!")
            return None

        print(f"\n[SAFE] Risk-aware path:")
        print(f"   {' -> '.join(safe_path)} (cost: {safe_cost:.2f})")

        # Step 3: Scan for dangerous segments
        print(f"\n[SCAN] Scanning route for dangers...")
        risky_edges = self._scan_for_danger(safe_path)

        # Step 4: Make decision
        if not risky_edges:
            decision = {
                'action': 'proceed',
                'path': safe_path,
                'cost': safe_cost,
                'shortest_path': shortest_path,
                'shortest_cost': shortest_cost,
                'message': '[OK] All segments below risk threshold. Safe to proceed!'
            }
        else:
            print(f"\n[WARNING] {len(risky_edges)} segment(s) above threshold!")
            decision = self._try_reroute(source, destination, risky_edges,
                                          shortest_path, shortest_cost)

        # Step 5: Store in memory
        self.memory.append({
            'source': source,
            'destination': destination,
            'action': decision['action'],
            'risky_segments': len(risky_edges),
            'conditions': dict(current_conditions),
        })

        return decision

    def _scan_for_danger(self, path):
        """Check each edge in the path for risk above threshold."""
        risky = []
        for i in range(len(path) - 1):
            node_a, node_b = path[i], path[i + 1]
            metadata = get_edge_metadata(node_a, node_b)
            risk = predict_edge_risk(node_a, node_b, metadata)

            status = "!! DANGER" if risk > self.risk_threshold else "OK"
            print(f"   {node_a} -> {node_b}: risk={risk:.3f} "
                  f"[{metadata['road_type']}, {metadata['speed_limit']}km/h] {status}")

            if risk > self.risk_threshold:
                risky.append((node_a, node_b, risk))

        return risky

    def _try_reroute(self, source, destination, risky_edges, shortest_path, shortest_cost):
        """Attempt to find a path that avoids all risky edges."""
        print(f"\n[REROUTE] Attempting reroute (removing {len(risky_edges)} edge(s))...")

        # Create a modified graph without risky edges
        safe_graph = copy.deepcopy(self.graph)
        for node_a, node_b, risk in risky_edges:
            safe_graph[node_a] = [(n, d) for n, d in safe_graph[node_a] if n != node_b]
            safe_graph[node_b] = [(n, d) for n, d in safe_graph[node_b] if n != node_a]

        # Try A* on the safe graph
        alt_path, alt_cost = a_star(
            safe_graph, self.positions, source, destination,
            weight_fn=self.weight_fn
        )

        if alt_path:
            avoided = [(a, b) for a, b, r in risky_edges]
            print(f"   >> Alternative found: {' -> '.join(alt_path)} (cost: {alt_cost:.2f})")
            return {
                'action': 'reroute',
                'path': alt_path,
                'cost': alt_cost,
                'avoided_edges': avoided,
                'shortest_path': shortest_path,
                'shortest_cost': shortest_cost,
                'message': f'>> Rerouted to avoid {len(risky_edges)} dangerous segment(s).'
            }
        else:
            # No safe alternative exists — must proceed carefully
            orig_path, orig_cost = a_star(
                self.graph, self.positions, source, destination,
                weight_fn=self.weight_fn
            )
            print(f"   [WARNING] No alternative available. Must proceed with caution.")
            return {
                'action': 'proceed_with_caution',
                'path': orig_path,
                'cost': orig_cost,
                'warnings': [(a, b, r) for a, b, r in risky_edges],
                'shortest_path': shortest_path,
                'shortest_cost': shortest_cost,
                'message': '[CAUTION] No safe alternative exists. Proceed with caution on flagged segments.'
            }

    def show_memory(self):
        """Display all past decisions."""
        print(f"\n[MEMORY] Agent Memory ({len(self.memory)} decision(s)):")
        for i, mem in enumerate(self.memory):
            cond = mem['conditions']
            print(f"   [{i+1}] {mem['source']}->{mem['destination']}: "
                  f"{mem['action'].upper()} | "
                  f"risky={mem['risky_segments']} | "
                  f"weather={cond['weather']}, hour={cond['hour_of_day']}")
