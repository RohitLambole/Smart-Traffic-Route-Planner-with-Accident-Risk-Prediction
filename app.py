"""
============================================================
  Streamlit Web UI — Smart Traffic Route Planner
============================================================
  Run: streamlit run app.py
============================================================
"""

import streamlit as st
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ── path setup ─────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from graph import city_graph, node_positions
from risk_routing import load_model, set_conditions, predict_edge_risk, current_conditions
from agent import TrafficAgent
from graph import get_edge_metadata

# ── page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Traffic Route Planner",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .stApp { background-color: #0f0f1a; }
    h1 { color: #00ff88 !important; font-family: 'Segoe UI', sans-serif; }
    h2, h3 { color: #e0e0e0 !important; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 6px 0;
    }
    .decide-proceed {
        background: linear-gradient(135deg, #0d4d2e, #1a7a4a);
        border: 2px solid #00ff88;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: #00ff88;
    }
    .decide-reroute {
        background: linear-gradient(135deg, #4d2000, #7a3d00);
        border: 2px solid #f39c12;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: #f39c12;
    }
    .decide-caution {
        background: linear-gradient(135deg, #4d0000, #7a0000);
        border: 2px solid #e74c3c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: #e74c3c;
    }
    .risk-row {
        font-family: monospace;
        font-size: 15px;
        padding: 4px 0;
    }
    div[data-testid="stSidebar"] {
        background-color: #16213e;
        border-right: 1px solid #333;
    }
    .stSelectbox label, .stSlider label { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)

# ── load model once ────────────────────────────────────────
@st.cache_resource
def init_model():
    return load_model()

model_loaded = init_model()

# ── helpers ────────────────────────────────────────────────
ALL_NODES = sorted(city_graph.keys())
DAYS      = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

def get_decision_html(action):
    icons = {'proceed': '✅ PROCEED', 'reroute': '↪ REROUTE',
             'proceed_with_caution': '⚠️ PROCEED WITH CAUTION'}
    classes = {'proceed': 'decide-proceed', 'reroute': 'decide-reroute',
               'proceed_with_caution': 'decide-caution'}
    label = icons.get(action, action.upper())
    cls   = classes.get(action, 'decide-proceed')
    return f'<div class="{cls}">{label}</div>'

def draw_route_map(result):
    """Draw dark-themed route map and return matplotlib figure."""
    G = nx.Graph()
    for node, neighbors in city_graph.items():
        for neighbor, dist in neighbors:
            G.add_edge(node, neighbor, weight=dist)

    edge_risks = {}
    for u, v in G.edges():
        meta = get_edge_metadata(u, v)
        edge_risks[(u, v)] = predict_edge_risk(u, v, meta)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    for (u, v), risk in edge_risks.items():
        x = [node_positions[u][0], node_positions[v][0]]
        y = [node_positions[u][1], node_positions[v][1]]
        color = '#e74c3c' if risk > 0.7 else ('#f39c12' if risk > 0.4 else '#2ecc71')
        alpha = 0.9 if risk > 0.7 else 0.7
        ax.plot(x, y, color=color, linewidth=2.5, alpha=alpha, zorder=1)
        ax.text((x[0]+x[1])/2, (y[0]+y[1])/2, f'{risk:.2f}',
                fontsize=6.5, color='white', alpha=0.7,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.12', facecolor='#222', alpha=0.5))

    # shortest path (blue dashed)
    sp = result.get('shortest_path', [])
    for i in range(len(sp)-1):
        u, v = sp[i], sp[i+1]
        x = [node_positions[u][0], node_positions[v][0]]
        y = [node_positions[u][1], node_positions[v][1]]
        ax.plot(x, y, color='#3498db', linewidth=5, linestyle='--', alpha=0.7, zorder=2)

    # chosen path (green solid)
    cp = result.get('path', [])
    for i in range(len(cp)-1):
        u, v = cp[i], cp[i+1]
        x = [node_positions[u][0], node_positions[v][0]]
        y = [node_positions[u][1], node_positions[v][1]]
        ax.plot(x, y, color='#00ff88', linewidth=5, alpha=0.95, zorder=3)

    # nodes
    for node, (x, y) in node_positions.items():
        if cp and node == cp[0]:
            color, size = '#00ff88', 22
        elif cp and node == cp[-1]:
            color, size = '#ff6b6b', 22
        else:
            color, size = '#e0e0e0', 16
        circle = plt.Circle((x, y), 0.32, color=color, zorder=4, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, node, fontsize=size, color='#1a1a2e',
                ha='center', va='center', fontweight='bold', zorder=5)

    legend = [
        mpatches.Patch(color='#2ecc71', label='Low Risk (< 0.4)'),
        mpatches.Patch(color='#f39c12', label='Medium Risk (0.4-0.7)'),
        mpatches.Patch(color='#e74c3c', label='High Risk (> 0.7)'),
        mpatches.Patch(color='#3498db', label=f'Shortest Path (cost: {result.get("shortest_cost",0):.1f})'),
        mpatches.Patch(color='#00ff88', label=f'Chosen Path (cost: {result.get("cost",0):.1f})'),
    ]
    ax.legend(handles=legend, loc='upper left', fontsize=9,
              facecolor='#16213e', edgecolor='#555', labelcolor='white')
    ax.set_xlim(-1.5, 13); ax.set_ylim(-1.5, 11)
    ax.set_aspect('equal'); ax.axis('off')
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════
#  SIDEBAR — INPUTS
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🗺️ Route Settings")
    st.markdown("---")

    source = st.selectbox("Source Node", ALL_NODES, index=1, key="src")
    dest   = st.selectbox("Destination Node",
                          [n for n in ALL_NODES if n != source], index=12, key="dst")

    st.markdown("---")
    st.markdown("## 🌤️ Conditions")

    hour    = st.slider("Hour of Day", 0, 23, 14, key="hour",
                        help="0 = midnight, 14 = 2 PM, 23 = 11 PM")
    day     = st.selectbox("Day of Week", DAYS, index=3, key="day")
    weather = st.selectbox("Weather", ["clear", "rain", "fog"], key="weather")
    traffic = st.slider("Traffic Density", 0.1, 1.0, 0.5, step=0.05, key="traffic",
                        help="0.1 = empty roads, 1.0 = gridlock")
    threshold = st.slider("Risk Threshold", 0.5, 0.9, 0.7, step=0.05, key="thresh",
                          help="Edges above this risk trigger rerouting")

    st.markdown("---")
    run_btn = st.button("🚀 Find Safe Route", use_container_width=True, type="primary")

# ══════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════
st.markdown("# 🛣️ Smart Traffic Route Planner")
st.markdown("### with Accident Risk Prediction · AI + ML Mini Project")
st.markdown("---")

if not model_loaded:
    st.error("Model not found! Run `python src/model.py` first.")
    st.stop()

# ── placeholder before running ──────────────────────────────
if not run_btn:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="metric-card">
            <h4 style="color:#00ff88">ML Model</h4>
            <p style="color:#aaa">Gradient Boosting<br>R² = 0.9177</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <h4 style="color:#3498db">Algorithm</h4>
            <p style="color:#aaa">A* Search<br>Risk-Aware Cost Function</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <h4 style="color:#f39c12">Agent</h4>
            <p style="color:#aaa">Goal-Based + Memory<br>Dynamic Rerouting</p>
        </div>""", unsafe_allow_html=True)

    st.info("Set your source, destination and conditions in the sidebar, then click **Find Safe Route**.")

    # show pre-generated graphs
    st.markdown("---")
    st.markdown("## 📊 Model Performance Analysis")
    g1 = os.path.join(os.path.dirname(__file__), 'graph_model_comparison.png')
    g2 = os.path.join(os.path.dirname(__file__), 'graph_feature_importance.png')
    g3 = os.path.join(os.path.dirname(__file__), 'graph_risk_distribution.png')
    if os.path.exists(g1): st.image(g1, use_container_width=True)
    c1, c2 = st.columns(2)
    if os.path.exists(g2): c1.image(g2, use_container_width=True)
    if os.path.exists(g3): c2.image(g3, use_container_width=True)

# ── run the simulation ──────────────────────────────────────
if run_btn:
    if source == dest:
        st.error("Source and Destination must be different!")
        st.stop()

    set_conditions(hour=hour, day=DAYS.index(day), weather=weather, traffic=traffic)

    with st.spinner("Planning route..."):
        agent  = TrafficAgent(city_graph, node_positions, risk_threshold=threshold)
        result = agent.plan_route(source, dest)

    if result is None:
        st.error("No path found between selected nodes.")
        st.stop()

    # ── top metrics ────────────────────────────────────────
    st.markdown("## 📍 Route Result")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Source",      source)
    m2.metric("Destination", dest)
    m3.metric("Route Cost",  f"{result['cost']:.2f}")
    m4.metric("Shortest Cost", f"{result.get('shortest_cost', 0):.2f}")

    # ── decision banner ────────────────────────────────────
    st.markdown(get_decision_html(result['action']), unsafe_allow_html=True)
    st.markdown(f"**{result['message']}**")
    st.markdown("---")

    # ── two-column: map + scan ─────────────────────────────
    left, right = st.columns([2, 1])

    with left:
        st.markdown("### 🗺️ Route Map")
        fig = draw_route_map(result)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("### 🔍 Edge Risk Scan")

        chosen_path = result['path']
        for i in range(len(chosen_path)-1):
            u, v    = chosen_path[i], chosen_path[i+1]
            meta    = get_edge_metadata(u, v)
            risk    = predict_edge_risk(u, v, meta)
            is_danger = risk > threshold

            color  = "#e74c3c" if is_danger else ("#f39c12" if risk > 0.4 else "#2ecc71")
            status = "!! DANGER" if is_danger else "OK"
            st.markdown(
                f'<div class="risk-row" style="color:{color}">'
                f'  {u} → {v} &nbsp;|&nbsp; risk = <b>{risk:.3f}</b> '
                f'&nbsp;|&nbsp; {meta["road_type"]} &nbsp;|&nbsp; <b>{status}</b>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### 📋 Conditions")
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Hour | {hour}:00 |
| Day | {day} |
| Weather | {weather.capitalize()} |
| Traffic | {int(traffic*100)}% |
| Risk Threshold | {threshold} |
""")

        if result.get('avoided_edges'):
            st.markdown("### 🚫 Avoided Edges")
            for a, b in result['avoided_edges']:
                st.markdown(f"- **{a} → {b}** (too dangerous)")

    # ── path comparison ────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔁 Path Comparison")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("**Normal GPS (Distance Only)**")
        sp = result.get('shortest_path', [])
        st.code(" → ".join(sp) + f"\nCost: {result.get('shortest_cost',0):.2f}", language=None)
    with pc2:
        st.markdown("**Our System (Risk-Aware)**")
        st.code(" → ".join(result['path']) + f"\nCost: {result['cost']:.2f}", language=None)

    # ── cost function ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚙️ Cost Function Used")
    st.latex(r"w(u,v) = \alpha \times d(u,v) + \beta \times (r(u,v) \times S)")
    st.markdown("""
| Parameter | Value | Meaning |
|-----------|-------|---------|
| α (alpha) | 0.6 | Distance importance |
| β (beta) | 0.4 | Safety importance |
| S (scale) | 10 | Normalizes risk to match distance units |
""")

    # ── result graphs ──────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 ML Model Analysis")
    g1 = os.path.join(os.path.dirname(__file__), 'graph_model_comparison.png')
    g2 = os.path.join(os.path.dirname(__file__), 'graph_feature_importance.png')
    g3 = os.path.join(os.path.dirname(__file__), 'graph_risk_distribution.png')
    if os.path.exists(g1): st.image(g1, caption="Model Comparison", use_container_width=True)
    c1, c2 = st.columns(2)
    if os.path.exists(g2): c1.image(g2, caption="Feature Importance", use_container_width=True)
    if os.path.exists(g3): c2.image(g3, caption="Risk Distribution", use_container_width=True)

# ── footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#555;'>Smart Traffic Route Planner · "
    "3rd Year CE Mini Project · AI + ML</p>",
    unsafe_allow_html=True
)
