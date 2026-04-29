"""
=============================================================
  MODULE 1: City Graph Definition
=============================================================
  This file defines:
    1. city_graph  — adjacency list (node → neighbors + distance)
    2. node_positions — (x, y) coordinates for heuristic + visualization
    3. edge_info — metadata per road segment (for risk prediction)

  The city has 15 nodes (A–O) and ~22 edges, resembling a
  small city grid with highways, residential roads, and lanes.
=============================================================
"""

# ---------------------------------------------------------------
# 1. CITY GRAPH — adjacency list
#    Format: { node: [(neighbor, distance_km), ...] }
# ---------------------------------------------------------------
city_graph = {
    'A': [('B', 3.0), ('C', 4.5)],
    'B': [('A', 3.0), ('D', 2.5), ('E', 5.0)],
    'C': [('A', 4.5), ('F', 3.5), ('G', 6.0)],
    'D': [('B', 2.5), ('E', 1.5), ('H', 4.0)],
    'E': [('B', 5.0), ('D', 1.5), ('F', 2.0), ('I', 3.0)],
    'F': [('C', 3.5), ('E', 2.0), ('G', 2.5), ('J', 4.5)],
    'G': [('C', 6.0), ('F', 2.5), ('K', 3.0)],
    'H': [('D', 4.0), ('I', 2.0), ('L', 3.5)],
    'I': [('E', 3.0), ('H', 2.0), ('J', 2.5), ('M', 4.0)],
    'J': [('F', 4.5), ('I', 2.5), ('K', 3.0), ('N', 3.5)],
    'K': [('G', 3.0), ('J', 3.0), ('O', 5.0)],
    'L': [('H', 3.5), ('M', 2.0)],
    'M': [('I', 4.0), ('L', 2.0), ('N', 2.5)],
    'N': [('J', 3.5), ('M', 2.5), ('O', 3.0)],
    'O': [('K', 5.0), ('N', 3.0)],
}

# ---------------------------------------------------------------
# 2. NODE POSITIONS — (x, y) for heuristic + visualization
#    Arranged roughly in a grid pattern
# ---------------------------------------------------------------
node_positions = {
    'A': (0, 6),
    'B': (3, 8),
    'C': (0, 3),
    'D': (5, 9),
    'E': (5, 6),
    'F': (3, 3),
    'G': (0, 0),
    'H': (8, 9),
    'I': (8, 6),
    'J': (6, 3),
    'K': (3, 0),
    'L': (11, 9),
    'M': (11, 6),
    'N': (9, 3),
    'O': (6, 0),
}

# ---------------------------------------------------------------
# 3. EDGE INFO — metadata for each road (used by risk model)
#    Only store in one direction; code handles reverse lookup.
# ---------------------------------------------------------------
edge_info = {
    # --- Highways (fast, higher base risk) ---
    ('A', 'B'): {'road_type': 'highway',     'speed_limit': 80, 'has_signal': 0, 'num_lanes': 4},
    ('B', 'E'): {'road_type': 'highway',     'speed_limit': 80, 'has_signal': 0, 'num_lanes': 4},
    ('D', 'H'): {'road_type': 'highway',     'speed_limit': 80, 'has_signal': 0, 'num_lanes': 4},
    ('H', 'L'): {'road_type': 'highway',     'speed_limit': 60, 'has_signal': 0, 'num_lanes': 2},

    # --- Main Roads (medium speed, usually have signals) ---
    ('B', 'D'): {'road_type': 'main_road',   'speed_limit': 60, 'has_signal': 1, 'num_lanes': 2},
    ('E', 'I'): {'road_type': 'main_road',   'speed_limit': 60, 'has_signal': 1, 'num_lanes': 2},
    ('I', 'M'): {'road_type': 'main_road',   'speed_limit': 60, 'has_signal': 1, 'num_lanes': 2},
    ('F', 'J'): {'road_type': 'main_road',   'speed_limit': 60, 'has_signal': 1, 'num_lanes': 2},
    ('J', 'N'): {'road_type': 'main_road',   'speed_limit': 40, 'has_signal': 1, 'num_lanes': 2},
    ('N', 'O'): {'road_type': 'main_road',   'speed_limit': 40, 'has_signal': 1, 'num_lanes': 2},

    # --- Residential Roads (slower, safer) ---
    ('A', 'C'): {'road_type': 'residential', 'speed_limit': 30, 'has_signal': 1, 'num_lanes': 2},
    ('C', 'F'): {'road_type': 'residential', 'speed_limit': 30, 'has_signal': 1, 'num_lanes': 2},
    ('C', 'G'): {'road_type': 'residential', 'speed_limit': 30, 'has_signal': 0, 'num_lanes': 1},
    ('E', 'F'): {'road_type': 'residential', 'speed_limit': 30, 'has_signal': 1, 'num_lanes': 2},
    ('D', 'E'): {'road_type': 'residential', 'speed_limit': 30, 'has_signal': 1, 'num_lanes': 2},
    ('I', 'J'): {'road_type': 'residential', 'speed_limit': 30, 'has_signal': 1, 'num_lanes': 2},
    ('L', 'M'): {'road_type': 'residential', 'speed_limit': 30, 'has_signal': 1, 'num_lanes': 2},
    ('M', 'N'): {'road_type': 'residential', 'speed_limit': 30, 'has_signal': 1, 'num_lanes': 2},

    # --- Lanes (narrow, slow) ---
    ('G', 'K'): {'road_type': 'lane',        'speed_limit': 20, 'has_signal': 0, 'num_lanes': 1},
    ('F', 'G'): {'road_type': 'lane',        'speed_limit': 20, 'has_signal': 0, 'num_lanes': 1},
    ('K', 'O'): {'road_type': 'lane',        'speed_limit': 20, 'has_signal': 0, 'num_lanes': 1},
    ('H', 'I'): {'road_type': 'residential', 'speed_limit': 40, 'has_signal': 1, 'num_lanes': 2},
    ('I', 'H'): {'road_type': 'residential', 'speed_limit': 40, 'has_signal': 1, 'num_lanes': 2},
    ('J', 'K'): {'road_type': 'lane',        'speed_limit': 20, 'has_signal': 0, 'num_lanes': 1},
    ('K', 'J'): {'road_type': 'lane',        'speed_limit': 20, 'has_signal': 0, 'num_lanes': 1},
}


def get_edge_metadata(node_a, node_b):
    """Look up edge metadata, handling both directions."""
    if (node_a, node_b) in edge_info:
        return edge_info[(node_a, node_b)]
    elif (node_b, node_a) in edge_info:
        return edge_info[(node_b, node_a)]
    else:
        # Default metadata for edges not explicitly defined
        return {'road_type': 'residential', 'speed_limit': 30, 'has_signal': 1, 'num_lanes': 2}


if __name__ == '__main__':
    print(f"City has {len(city_graph)} nodes and", end=" ")
    edge_count = sum(len(v) for v in city_graph.values()) // 2
    print(f"{edge_count} edges")
    print(f"\nNodes: {', '.join(sorted(city_graph.keys()))}")
    print(f"\nSample connections from 'A':")
    for neighbor, dist in city_graph['A']:
        meta = get_edge_metadata('A', neighbor)
        print(f"  A → {neighbor}: {dist} km | {meta['road_type']} | {meta['speed_limit']} km/h")
