"""
=============================================================
  MODULE 6: Visualization
=============================================================
  Draws the city graph with:
    - Nodes as labeled circles
    - Edges colored by risk level (green/orange/red)
    - Shortest path (blue dashed)
    - Safe path (green solid)

  Uses matplotlib + networkx for drawing.
=============================================================
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from graph import get_edge_metadata
from risk_routing import predict_edge_risk, load_model


def visualize_routes(graph, positions, result, save_path='route_map.png'):
    """
    Visualize the city graph with risk-colored edges and route comparison.

    Parameters:
        graph     — adjacency list
        positions — {node: (x, y)}
        result    — dict from agent.plan_route() with 'path', 'shortest_path', etc.
        save_path — where to save the image
    """
    # Ensure model is loaded
    load_model()

    # Build networkx graph
    G = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor, dist in neighbors:
            G.add_edge(node, neighbor, weight=dist)

    # Calculate risk for each edge
    edge_risks = {}
    for u, v in G.edges():
        metadata = get_edge_metadata(u, v)
        risk = predict_edge_risk(u, v, metadata)
        edge_risks[(u, v)] = risk

    # ---- Setup figure ----
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # ---- Draw edges colored by risk ----
    for (u, v), risk in edge_risks.items():
        x_vals = [positions[u][0], positions[v][0]]
        y_vals = [positions[u][1], positions[v][1]]

        if risk > 0.7:
            color = '#e74c3c'   # Red — dangerous
            alpha = 0.9
        elif risk > 0.4:
            color = '#f39c12'   # Orange — moderate
            alpha = 0.7
        else:
            color = '#2ecc71'   # Green — safe
            alpha = 0.6

        ax.plot(x_vals, y_vals, color=color, linewidth=2.5, alpha=alpha, zorder=1)

        # Risk label on edge
        mid_x = (x_vals[0] + x_vals[1]) / 2
        mid_y = (y_vals[0] + y_vals[1]) / 2
        ax.text(mid_x, mid_y, f'{risk:.2f}', fontsize=7, color='white',
                alpha=0.6, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#333', alpha=0.5))

    # ---- Draw shortest path (blue dashed) ----
    shortest_path = result.get('shortest_path')
    if shortest_path:
        for i in range(len(shortest_path) - 1):
            u, v = shortest_path[i], shortest_path[i + 1]
            x_vals = [positions[u][0], positions[v][0]]
            y_vals = [positions[u][1], positions[v][1]]
            ax.plot(x_vals, y_vals, color='#3498db', linewidth=4,
                    linestyle='--', alpha=0.8, zorder=2)

    # ---- Draw safe/chosen path (bright green solid) ----
    chosen_path = result.get('path')
    if chosen_path:
        for i in range(len(chosen_path) - 1):
            u, v = chosen_path[i], chosen_path[i + 1]
            x_vals = [positions[u][0], positions[v][0]]
            y_vals = [positions[u][1], positions[v][1]]
            ax.plot(x_vals, y_vals, color='#00ff88', linewidth=4,
                    alpha=0.9, zorder=3)

    # ---- Draw nodes ----
    for node, (x, y) in positions.items():
        # Highlight start and end
        if chosen_path and node == chosen_path[0]:
            node_color = '#00ff88'
            node_size = 22
        elif chosen_path and node == chosen_path[-1]:
            node_color = '#ff6b6b'
            node_size = 22
        else:
            node_color = '#e0e0e0'
            node_size = 18

        circle = plt.Circle((x, y), 0.35, color=node_color, zorder=4, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, node, fontsize=node_size, color='#1a1a2e',
                ha='center', va='center', fontweight='bold', zorder=5)

    # ---- Legend ----
    legend_items = [
        mpatches.Patch(color='#2ecc71', label='Low Risk (< 0.4)'),
        mpatches.Patch(color='#f39c12', label='Medium Risk (0.4-0.7)'),
        mpatches.Patch(color='#e74c3c', label='High Risk (> 0.7)'),
        mpatches.Patch(color='#3498db', label=f'Shortest Path (cost: {result.get("shortest_cost", 0):.1f})'),
        mpatches.Patch(color='#00ff88', label=f'Chosen Path (cost: {result.get("cost", 0):.1f})'),
    ]
    ax.legend(handles=legend_items, loc='upper left', fontsize=10,
              facecolor='#16213e', edgecolor='#e0e0e0', labelcolor='white')

    # ---- Title ----
    action = result.get('action', 'unknown').upper()
    ax.set_title(f'Smart Traffic Route Planner\n'
                 f'Decision: {action} | {result.get("message", "")}',
                 fontsize=14, color='white', fontweight='bold', pad=20)

    ax.set_xlim(-1.5, 13)
    ax.set_ylim(-1.5, 11)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    print(f"\n[OK] Map saved to: {save_path}")
    plt.show()
