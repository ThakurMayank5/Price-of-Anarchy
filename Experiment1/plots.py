"""
Visualization Module for Traffic Assignment Experiments
=======================================================

Reads results.csv, convergence.csv, and edge_flows.csv to produce:
  1. PoA vs Graph Size (by graph type)
  2. PoA by Graph Type (grouped bar)
  3. Convergence Curves
  4. Runtime vs Nodes
  5. PoA vs Density
  6. Nash vs Social Cost comparison
  7. Edge Flow Distribution (histogram)
  8. Top-N loaded edges (Nash vs Social bar chart)
  9. Nash vs Social edge flow scatter
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────

COLORS = {
    "grid": "#2196F3",
    "erdos_renyi": "#FF9800",
    "scale_free": "#4CAF50",
    "nash": "#E53935",
    "social": "#1E88E5",
    "linear": "#7E57C2",
    "bpr": "#26A69A",
}

def apply_style():
    """Apply a clean plot style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })


# ─────────────────────────────────────────────────────────────────────
# Plot Functions
# ─────────────────────────────────────────────────────────────────────

def plot_poa_vs_graph_size(df, plots_dir):
    """Plot 1: PoA vs number of nodes, colored by graph type."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for gtype in df["graph_type"].unique():
        subset = df[df["graph_type"] == gtype]
        grouped = subset.groupby("num_nodes")["poa"].mean().reset_index()
        color = COLORS.get(gtype, "#999999")
        ax.plot(grouped["num_nodes"], grouped["poa"], "o-",
                label=gtype.replace("_", " ").title(), color=color,
                markersize=7, linewidth=2)

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Price of Anarchy (PoA)")
    ax.set_title("Price of Anarchy vs Graph Size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "poa_vs_graph_size.png"))
    plt.close(fig)
    print("  Saved: poa_vs_graph_size.png")


def plot_poa_by_graph_type(df, plots_dir):
    """Plot 2: Grouped bar chart of PoA by graph type and latency."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    graph_types = df["graph_type"].unique()
    latency_types = df["latency_type"].unique()
    x = np.arange(len(graph_types))
    width = 0.35

    for i, lt in enumerate(latency_types):
        means = []
        for gt in graph_types:
            subset = df[(df["graph_type"] == gt) & (df["latency_type"] == lt)]
            means.append(subset["poa"].mean() if len(subset) > 0 else 0)
        color = COLORS.get(lt, "#999999")
        ax.bar(x + i * width, means, width, label=lt.upper(), color=color, alpha=0.85)

    ax.set_xlabel("Graph Type")
    ax.set_ylabel("Average PoA")
    ax.set_title("Price of Anarchy by Graph Type and Latency")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([gt.replace("_", " ").title() for gt in graph_types])
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "poa_by_graph_type.png"))
    plt.close(fig)
    print("  Saved: poa_by_graph_type.png")


def plot_convergence_curves(conv_df, plots_dir):
    """Plot 3: Convergence curves (relative gap vs iteration) for select experiments."""
    apply_style()

    # Pick a representative subset: first experiment per graph type for Nash
    results_ids = conv_df["experiment_id"].unique()
    # Group by graph type (parse from experiment_id)
    selected = []
    seen_types = set()
    for eid in results_ids:
        parts = eid.split("_")
        # Extract graph type from experiment id
        if "grid" in eid:
            gtype = "grid"
        elif "er" in eid:
            gtype = "erdos_renyi"
        elif "sf" in eid:
            gtype = "scale_free"
        else:
            gtype = "unknown"
        key = gtype
        if key not in seen_types:
            seen_types.add(key)
            selected.append(eid)
        if len(selected) >= 6:
            break

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for solver_type, ax in zip(["nash", "social"], axes):
        for eid in selected:
            subset = conv_df[
                (conv_df["experiment_id"] == eid) &
                (conv_df["solver_type"] == solver_type)
            ]
            if len(subset) == 0:
                continue
            label = eid.replace("_", " ")[:30]
            ax.plot(subset["iteration"], subset["relative_gap"], linewidth=1.5,
                    label=label, alpha=0.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Relative Gap")
        ax.set_title(f"Convergence — {solver_type.title()} Equilibrium")
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "convergence_curves.png"))
    plt.close(fig)
    print("  Saved: convergence_curves.png")


def plot_runtime_vs_nodes(df, plots_dir):
    """Plot 4: Runtime vs number of nodes."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for gtype in df["graph_type"].unique():
        subset = df[df["graph_type"] == gtype]
        grouped = subset.groupby("num_nodes")["runtime"].mean().reset_index()
        color = COLORS.get(gtype, "#999999")
        ax.plot(grouped["num_nodes"], grouped["runtime"], "s-",
                label=gtype.replace("_", " ").title(), color=color,
                markersize=7, linewidth=2)

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Computation Time vs Graph Size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "runtime_vs_nodes.png"))
    plt.close(fig)
    print("  Saved: runtime_vs_nodes.png")


def plot_poa_vs_density(df, plots_dir):
    """Plot 5: PoA vs edge density, colored by graph type."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for gtype in df["graph_type"].unique():
        subset = df[df["graph_type"] == gtype]
        color = COLORS.get(gtype, "#999999")
        ax.scatter(subset["density"], subset["poa"],
                   label=gtype.replace("_", " ").title(), color=color,
                   alpha=0.6, s=50, edgecolors="white", linewidth=0.5)

    ax.set_xlabel("Edge Density")
    ax.set_ylabel("Price of Anarchy (PoA)")
    ax.set_title("PoA vs Network Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "poa_vs_density.png"))
    plt.close(fig)
    print("  Saved: poa_vs_density.png")


def plot_nash_vs_social_cost(df, plots_dir):
    """Plot 6: Nash cost vs Social cost comparison."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for gtype in df["graph_type"].unique():
        subset = df[df["graph_type"] == gtype]
        color = COLORS.get(gtype, "#999999")
        ax.scatter(subset["optimal_cost"], subset["equilibrium_cost"],
                   label=gtype.replace("_", " ").title(), color=color,
                   alpha=0.6, s=50, edgecolors="white", linewidth=0.5)

    # Add y=x reference line (where PoA = 1)
    all_costs = pd.concat([df["equilibrium_cost"], df["optimal_cost"]])
    max_val = all_costs.max() * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="PoA = 1 (ideal)")

    ax.set_xlabel("Social Optimum Cost")
    ax.set_ylabel("Nash Equilibrium Cost")
    ax.set_title("Nash vs Social Optimum Cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "nash_vs_social_cost.png"))
    plt.close(fig)
    print("  Saved: nash_vs_social_cost.png")


def plot_poa_vs_agents(df, plots_dir):
    """Plot 7: PoA vs number of agents, by graph type."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for gtype in df["graph_type"].unique():
        subset = df[df["graph_type"] == gtype]
        grouped = subset.groupby("num_agents")["poa"].mean().reset_index()
        color = COLORS.get(gtype, "#999999")
        ax.plot(grouped["num_agents"], grouped["poa"], "D-",
                label=gtype.replace("_", " ").title(), color=color,
                markersize=6, linewidth=2)

    ax.set_xlabel("Number of Agents (Demand)")
    ax.set_ylabel("Average PoA")
    ax.set_title("Price of Anarchy vs Traffic Demand")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "poa_vs_agents.png"))
    plt.close(fig)
    print("  Saved: poa_vs_agents.png")


# ─────────────────────────────────────────────────────────────────────
# Edge-Level Visualizations
# ─────────────────────────────────────────────────────────────────────

def plot_edge_flow_scatter(edge_df, plots_dir):
    """Plot 8: Scatter of Nash flow vs Social flow per edge."""
    apply_style()

    # Pick a representative experiment (medium-sized)
    experiment_ids = edge_df["experiment_id"].unique()
    if len(experiment_ids) == 0:
        return

    # Select up to 4 experiments from different graph types
    selected = []
    seen = set()
    for eid in experiment_ids:
        if "grid" in eid:
            key = "grid"
        elif "er" in eid:
            key = "er"
        elif "sf" in eid:
            key = "sf"
        else:
            key = "other"
        if key not in seen:
            seen.add(key)
            selected.append(eid)
        if len(selected) >= 3:
            break

    fig, axes = plt.subplots(1, len(selected), figsize=(6 * len(selected), 5))
    if len(selected) == 1:
        axes = [axes]

    for ax, eid in zip(axes, selected):
        subset = edge_df[edge_df["experiment_id"] == eid]
        ax.scatter(subset["social_flow"], subset["nash_flow"],
                   alpha=0.4, s=20, color="#1976D2", edgecolors="white", linewidth=0.3)
        max_val = max(subset["nash_flow"].max(), subset["social_flow"].max()) * 1.1
        if max_val > 0:
            ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3)
        ax.set_xlabel("Social Optimum Flow")
        ax.set_ylabel("Nash Equilibrium Flow")
        ax.set_title(eid.replace("_", " ")[:35], fontsize=10)

    fig.suptitle("Edge-Level: Nash vs Social Flow", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "edge_flow_scatter.png"))
    plt.close(fig)
    print("  Saved: edge_flow_scatter.png")


def plot_edge_flow_distribution(edge_df, plots_dir):
    """Plot 9: Histogram of edge flows (Nash vs Social)."""
    apply_style()

    # Pick one representative experiment per graph type
    selected = []
    seen = set()
    for eid in edge_df["experiment_id"].unique():
        if "grid" in eid:
            key = "grid"
        elif "er" in eid:
            key = "er"
        elif "sf" in eid:
            key = "sf"
        else:
            key = "other"
        if key not in seen:
            seen.add(key)
            selected.append(eid)
        if len(selected) >= 3:
            break

    fig, axes = plt.subplots(1, len(selected), figsize=(6 * len(selected), 5))
    if len(selected) == 1:
        axes = [axes]

    for ax, eid in zip(axes, selected):
        subset = edge_df[edge_df["experiment_id"] == eid]
        # Only show edges with flow > 0
        nash_flows = subset["nash_flow"][subset["nash_flow"] > 1e-6]
        social_flows = subset["social_flow"][subset["social_flow"] > 1e-6]

        bins = 30
        if len(nash_flows) > 0:
            ax.hist(nash_flows, bins=bins, alpha=0.6, color=COLORS["nash"],
                    label="Nash", edgecolor="white", linewidth=0.5)
        if len(social_flows) > 0:
            ax.hist(social_flows, bins=bins, alpha=0.6, color=COLORS["social"],
                    label="Social", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Edge Flow")
        ax.set_ylabel("Count")
        ax.set_title(eid.replace("_", " ")[:35], fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Edge Flow Distribution: Nash vs Social", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "edge_flow_distribution.png"))
    plt.close(fig)
    print("  Saved: edge_flow_distribution.png")


def plot_top_loaded_edges(edge_df, plots_dir):
    """Plot 10: Top-N most loaded edges (Nash vs Social) bar chart."""
    apply_style()

    # Pick one experiment per graph type
    selected = []
    seen = set()
    for eid in edge_df["experiment_id"].unique():
        if "grid" in eid:
            key = "grid"
        elif "er" in eid:
            key = "er"
        elif "sf" in eid:
            key = "sf"
        else:
            key = "other"
        if key not in seen:
            seen.add(key)
            selected.append(eid)
        if len(selected) >= 3:
            break

    fig, axes = plt.subplots(len(selected), 1,
                             figsize=(12, 4 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for ax, eid in zip(axes, selected):
        subset = edge_df[edge_df["experiment_id"] == eid].copy()
        # Sort by Nash flow descending, take top 15
        subset = subset.nlargest(15, "nash_flow")
        edge_labels = [f"{r['edge_source']}->{r['edge_target']}"
                       for _, r in subset.iterrows()]
        x = np.arange(len(edge_labels))
        width = 0.35

        ax.barh(x - width/2, subset["nash_flow"], width,
                label="Nash", color=COLORS["nash"], alpha=0.85)
        ax.barh(x + width/2, subset["social_flow"], width,
                label="Social", color=COLORS["social"], alpha=0.85)
        ax.set_yticks(x)
        ax.set_yticklabels(edge_labels, fontsize=7)
        ax.set_xlabel("Flow")
        ax.set_title(f"Top Loaded Edges — {eid.replace('_', ' ')[:40]}", fontsize=10)
        ax.legend(fontsize=8)
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "top_loaded_edges.png"))
    plt.close(fig)
    print("  Saved: top_loaded_edges.png")


# ─────────────────────────────────────────────────────────────────────
# Master Plot Generator
# ─────────────────────────────────────────────────────────────────────

def generate_all_plots(results_dir):
    """Read CSVs and generate all plots."""
    print("\n========== Generating Plots ==========")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "results.csv")
    conv_path = os.path.join(results_dir, "convergence.csv")
    edge_path = os.path.join(results_dir, "edge_flows.csv")

    # Load data
    df = pd.read_csv(results_path)
    print(f"  Loaded results.csv: {len(df)} rows")

    # ── Results-based plots ──
    plot_poa_vs_graph_size(df, plots_dir)
    plot_poa_by_graph_type(df, plots_dir)
    plot_runtime_vs_nodes(df, plots_dir)
    plot_poa_vs_density(df, plots_dir)
    plot_nash_vs_social_cost(df, plots_dir)
    plot_poa_vs_agents(df, plots_dir)

    # ── Convergence plots ──
    if os.path.exists(conv_path):
        conv_df = pd.read_csv(conv_path)
        print(f"  Loaded convergence.csv: {len(conv_df)} rows")
        plot_convergence_curves(conv_df, plots_dir)

    # ── Edge-level plots ──
    if os.path.exists(edge_path):
        edge_df = pd.read_csv(edge_path)
        print(f"  Loaded edge_flows.csv: {len(edge_df)} rows")
        plot_edge_flow_scatter(edge_df, plots_dir)
        plot_edge_flow_distribution(edge_df, plots_dir)
        plot_top_loaded_edges(edge_df, plots_dir)

    print(f"\n  All plots saved to: {plots_dir}")
    print("========== Done ==========\n")
