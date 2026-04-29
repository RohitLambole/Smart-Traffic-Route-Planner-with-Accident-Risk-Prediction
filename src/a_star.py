"""
=============================================================
  MODULE 2: A* Search Algorithm
=============================================================
  Generic A* implementation with a pluggable weight function.

  - Default: uses base distance as weight
  - Custom: pass weight_fn(node_a, node_b, base_dist) → weight
    This is how we'll inject risk-aware routing later.
=============================================================
"""

import heapq
import math


def euclidean_distance(positions, node_a, node_b):
    """
    Heuristic function: straight-line distance between two nodes.

    WHY THIS IS ADMISSIBLE:
    Straight line is always ≤ road distance (roads aren't shorter
    than a straight line). So it never overestimates → A* stays optimal.
    """
    x1, y1 = positions[node_a]
    x2, y2 = positions[node_b]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def a_star(graph, positions, start, goal, weight_fn=None):
    """
    A* pathfinding algorithm.

    Parameters:
        graph       — adjacency list {node: [(neighbor, distance), ...]}
        positions   — {node: (x, y)} for heuristic calculation
        start       — starting node
        goal        — destination node
        weight_fn   — optional function(node_a, node_b, base_dist) → weight
                       If None, uses base distance directly.

    Returns:
        (path, total_cost)  — list of nodes and the cost
        (None, float('inf')) — if no path exists
    """
    # Priority queue: (f_score, node_name)
    open_set = [(0, start)]

    # Track where each node was reached from
    came_from = {}

    # g_score[n] = actual cost from start to n
    g_score = {start: 0}

    # Track which nodes we've fully processed
    closed_set = set()

    while open_set:
        current_f, current = heapq.heappop(open_set)

        # Skip if already processed (handles duplicate entries in heap)
        if current in closed_set:
            continue
        closed_set.add(current)

        # Reached the goal — reconstruct path
        if current == goal:
            path = _reconstruct_path(came_from, current, start)
            return path, g_score[goal]

        # Explore all neighbors
        for neighbor, base_distance in graph[current]:
            if neighbor in closed_set:
                continue

            # Calculate edge weight (pluggable!)
            if weight_fn:
                weight = weight_fn(current, neighbor, base_distance)
            else:
                weight = base_distance

            tentative_g = g_score[current] + weight

            # Only update if this path is better
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + euclidean_distance(positions, neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    # No path found
    return None, float('inf')


def _reconstruct_path(came_from, current, start):
    """Walk backwards through came_from to build the path."""
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# ---- Quick test ----
if __name__ == '__main__':
    # Import graph from our graph module
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from graph import city_graph, node_positions

    print("Testing A* (distance-only)...")
    path, cost = a_star(city_graph, node_positions, 'A', 'O')
    if path:
        print(f"  Path: {' → '.join(path)}")
        print(f"  Cost: {cost:.2f} km")
    else:
        print("  No path found!")

    print("\nTesting A* (A → L)...")
    path2, cost2 = a_star(city_graph, node_positions, 'A', 'L')
    if path2:
        print(f"  Path: {' → '.join(path2)}")
        print(f"  Cost: {cost2:.2f} km")
