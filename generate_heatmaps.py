"""
Week 7 — Congestion Heat Maps on Network Edges
================================================

Visualises how traffic flow distributes across edges at the Nash
equilibrium (selfish baseline) and under 100% bandit-agent populations
for all three algorithms (ε-Greedy, UCB, Thompson Sampling), on both
the Braess and Pigou networks.

Edge width and colour intensity both encode congestion level (flow count).
This makes visually clear which edges get over-used under selfish routing
and how each learning algorithm redistributes traffic.

Outputs (in results/week7/):
    congestion_heatmap_braess.png
    congestion_heatmap_pigou.png

Usage:
    python generate_heatmaps.py
    (must be run from repo root)
"""

import copy
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

# ── Path setup: make all four experiment modules importable ──
REPO_ROOT = Path(__file__).resolve().parent
for exp_dir in ["Experiment1", "Experiment2", "Experiment3", "Experiment4"]:
    sys.path.insert(0, str(REPO_ROOT / exp_dir))

from epsilon_greedy_agent import run_mixed_simulation as run_eg
from ucb_agent import run_mixed_simulation as run_ucb
from thompson_sampling_agent import run_mixed_simulation as run_ts

from braess_epsilon_greedy import build_braess_network, analytical_braess_costs
from pigou_epsilon_greedy import build_pigou_network, analytical_pigou_costs

OUT_DIR = REPO_ROOT / "results" / "week7"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_AGENTS = 50
N_STEPS = 2000  # longer run so flows stabilise before we snapshot them
EPSILON = 0.1


# ─────────────────────────────────────────────────────────────────────
# Flow extraction helpers
# ─────────────────────────────────────────────────────────────────────

def get_selfish_flow_braess(n_agents):
    """
    Analytical selfish (Nash) flow for Braess: all agents take the
    shortcut path s→a→b→t, so edges s→a, a→b, b→t carry full load.
    """
    return {
        (0, 1): n_agents,   # s → a
        (1, 2): n_agents,   # a → b  (shortcut)
        (2, 3): n_agents,   # b → t
        (1, 3): 0,           # a → t  (unused)
        (0, 2): 0,           # s → b  (unused)
    }


def get_selfish_flow_pigou(n_agents):
    """
    Analytical selfish (Nash) flow for Pigou: all agents take the
    congestion-dependent road (edge 0→1 in the two-link network),
    since at Nash equilibrium both routes have equal latency but
    the variable-cost route is what everyone piles onto.
    """
    return {
        (0, 1): n_agents,   # congestion-dependent road (variable cost)
        (0, 2): 0,           # constant-cost road (unused at Nash)
        (1, 3): n_agents,
        (2, 3): 0,
    }


def run_bandit_flow(run_fn, G, source, sink, opt_cost, label):
    """Run 100% bandit simulation and return final edge flow dict."""
    G_copy = copy.deepcopy(G)
    result = run_fn(
        G_copy, source, sink,
        n_agents=N_AGENTS,
        pct_bandit=1.0,
        n_steps=N_STEPS,
        opt_cost=opt_cost,
    )
    return result["final_flow_dict"]


# ─────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────

def draw_flow_graph(ax, G, flow_dict, pos, title, max_flow,
                    node_labels=None, edge_labels=None):
    """
    Draw a single network panel with edges coloured and sized by flow.

    Parameters
    ----------
    ax          : matplotlib Axes
    G           : nx.DiGraph
    flow_dict   : {(u,v): flow_count}
    pos         : {node: (x, y)} layout dict
    title       : panel title string
    max_flow    : normalisation ceiling (same across all panels in a figure)
    node_labels : optional dict of {node: label_string}
    edge_labels : optional dict of {(u,v): label_string}
    """
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis("off")

    cmap = cm.get_cmap("YlOrRd")

    edges = list(G.edges())
    flows = np.array([flow_dict.get(e, 0) for e in edges], dtype=float)
    norm_flows = flows / max_flow if max_flow > 0 else flows

    edge_colors = [cmap(v) for v in norm_flows]
    edge_widths = [1.0 + 6.0 * v for v in norm_flows]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500,
                           node_color="#2c3e50", linewidths=1.5,
                           edgecolors="#ecf0f1")
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels=node_labels or {n: str(n) for n in G.nodes()},
                            font_color="white", font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges,
                           edge_color=edge_colors, width=edge_widths,
                           arrows=True, arrowsize=18, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.08",
                           min_source_margin=18, min_target_margin=18)

    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels,
                                     font_size=8, label_pos=0.35,
                                     bbox=dict(boxstyle="round,pad=0.2",
                                               fc="white", alpha=0.7))

    # per-panel flow annotation on each edge
    for (u, v), flow in flow_dict.items():
        if (u, v) in pos or u in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.annotate(f"{int(flow)}",
                        xy=(mx, my), fontsize=8,
                        ha="center", va="center",
                        color="#2c3e50",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc="white", alpha=0.75, lw=0))


def add_colorbar(fig, cmap, max_flow, label="Edge flow (# agents)"):
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=0, vmax=int(max_flow)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.6, pad=0.02)
    cbar.set_label(label, fontsize=10)


# ─────────────────────────────────────────────────────────────────────
# Braess heat map
# ─────────────────────────────────────────────────────────────────────

def plot_braess_heatmap():
    G_braess, source, sink = build_braess_network(n_agents=N_AGENTS)
    _, opt_cost, _ = analytical_braess_costs(N_AGENTS)

    node_labels = {0: "s", 1: "a", 2: "b", 3: "t"}
    edge_labels = {
        (0, 1): "x/N",
        (1, 3): "1",
        (0, 2): "1",
        (2, 3): "x/N",
        (1, 2): "0 (shortcut)",
    }

    pos = {0: (0, 0.5), 1: (1, 1), 2: (1, 0), 3: (2, 0.5)}

    selfish_flow = get_selfish_flow_braess(N_AGENTS)
    eg_flow  = run_bandit_flow(run_eg,  copy.deepcopy(G_braess), source, sink, opt_cost, "ε-Greedy")
    ucb_flow = run_bandit_flow(run_ucb, copy.deepcopy(G_braess), source, sink, opt_cost, "UCB")
    ts_flow  = run_bandit_flow(run_ts,  copy.deepcopy(G_braess), source, sink, opt_cost, "Thompson")

    panels = [
        ("Selfish (Nash)\nAll agents on shortcut path", selfish_flow),
        ("ε-Greedy\n(100% bandit)", eg_flow),
        ("UCB\n(100% bandit)", ucb_flow),
        ("Thompson Sampling\n(100% bandit)", ts_flow),
    ]

    max_flow = N_AGENTS  # normalise to full population

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        "Congestion Heat Map — Braess Network\n"
        "Edge colour + width = traffic flow (darker/thicker = more congested)",
        fontsize=13, y=1.02,
    )

    cmap = cm.get_cmap("YlOrRd")

    for ax, (title, flow) in zip(axes, panels):
        draw_flow_graph(ax, G_braess, flow, pos, title, max_flow,
                        node_labels=node_labels, edge_labels=edge_labels)

    add_colorbar(fig, cmap, max_flow)
    fig.tight_layout()
    out_path = OUT_DIR / "congestion_heatmap_braess.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────
# Pigou heat map
# ─────────────────────────────────────────────────────────────────────

def plot_pigou_heatmap():
    G_pigou, source, sink = build_pigou_network(n_agents=N_AGENTS)
    _, opt_cost, _ = analytical_pigou_costs(N_AGENTS)

    node_labels = {0: "s", 1: "mid1", 2: "mid2", 3: "t"}
    edge_labels = {
        (0, 1): "x/N (variable)",
        (0, 2): "1 (fixed)",
        (1, 3): "",
        (2, 3): "",
    }

    pos = {0: (0, 0.5), 1: (1, 1), 2: (1, 0), 3: (2, 0.5)}

    selfish_flow = get_selfish_flow_pigou(N_AGENTS)
    eg_flow  = run_bandit_flow(run_eg,  copy.deepcopy(G_pigou), source, sink, opt_cost, "ε-Greedy")
    ucb_flow = run_bandit_flow(run_ucb, copy.deepcopy(G_pigou), source, sink, opt_cost, "UCB")
    ts_flow  = run_bandit_flow(run_ts,  copy.deepcopy(G_pigou), source, sink, opt_cost, "Thompson")

    panels = [
        ("Selfish (Nash)\nAll on variable-cost road", selfish_flow),
        ("ε-Greedy\n(100% bandit)", eg_flow),
        ("UCB\n(100% bandit)", ucb_flow),
        ("Thompson Sampling\n(100% bandit)", ts_flow),
    ]

    max_flow = N_AGENTS
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        "Congestion Heat Map — Pigou Network\n"
        "Edge colour + width = traffic flow (darker/thicker = more congested)",
        fontsize=13, y=1.02,
    )

    cmap = cm.get_cmap("YlOrRd")
    for ax, (title, flow) in zip(axes, panels):
        draw_flow_graph(ax, G_pigou, flow, pos, title, max_flow,
                        node_labels=node_labels, edge_labels=edge_labels)

    add_colorbar(fig, cmap, max_flow)
    fig.tight_layout()
    out_path = OUT_DIR / "congestion_heatmap_pigou.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    print("Generating Braess congestion heat map ...")
    plot_braess_heatmap()
    print("Generating Pigou congestion heat map ...")
    plot_pigou_heatmap()
    print("\nDone.")
