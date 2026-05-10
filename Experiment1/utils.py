"""
Utility Functions for Traffic Assignment Experiments
====================================================

Provides:
  - Graph generation (Grid, Erdos-Renyi, Scale-Free)
  - Latency parameter assignment (linear, BPR)
  - Source/target selection
  - CSV I/O for results, convergence, and edge flows
"""

import csv
import os
import random

import networkx as nx


# ─────────────────────────────────────────────────────────────────────
# Graph Generation
# ─────────────────────────────────────────────────────────────────────

def generate_grid_graph(rows, cols, seed=None):
    """
    Create a directed 2D grid graph.
    Nodes are labeled as (row, col) tuples.
    Source = (0,0), Target = (rows-1, cols-1).
    """
    graph = nx.grid_2d_graph(rows, cols).to_directed()
    return graph


def generate_erdos_renyi_graph(n, p, seed=None):
    """
    Create a directed Erdos-Renyi random graph.
    Retries with incremented seed if no s-t path exists.
    """
    max_tries = 20
    for attempt in range(max_tries):
        current_seed = seed + attempt if seed is not None else None
        graph = nx.erdos_renyi_graph(n, p, seed=current_seed, directed=True)
        # Check if there is at least one reachable pair
        source, target = pick_source_target_general(graph)
        if source is not None:
            return graph, source, target
    raise ValueError(
        f"Could not generate connected ER graph with n={n}, p={p} "
        f"after {max_tries} tries. Increase p."
    )


def generate_scale_free_graph(n, m, seed=None):
    """
    Create a directed Barabasi-Albert (scale-free) graph.
    BA graphs are always connected, so path always exists.
    """
    graph = nx.barabasi_albert_graph(n, m, seed=seed).to_directed()
    return graph


def pick_source_target_general(graph):
    """
    Pick source and target that maximize shortest path distance.
    Returns (source, target) or (None, None) if no path exists.
    """
    nodes = sorted(graph.nodes())
    # Try node 0 as source, find farthest reachable node
    for candidate_source in nodes:
        reachable = nx.descendants(graph, candidate_source)
        if not reachable:
            continue
        # Find farthest node by BFS
        lengths = nx.single_source_shortest_path_length(graph, candidate_source)
        # Remove source itself
        lengths.pop(candidate_source, None)
        if not lengths:
            continue
        farthest = max(lengths, key=lengths.get)
        return candidate_source, farthest
    return None, None


# ─────────────────────────────────────────────────────────────────────
# Latency Assignment
# ─────────────────────────────────────────────────────────────────────

def assign_latency(graph, latency_type, demand, seed=42):
    """
    Assign latency parameters to every edge.

    Linear:  l(f) = a*f + b
        a ~ Uniform(0.5, 2.0)
        b ~ Uniform(0.5, 2.0)

    BPR:     l(f) = free_flow * (1 + 0.15 * (f/capacity)^4)
        free_flow ~ Uniform(1.0, 5.0)
        capacity  ~ demand * Uniform(0.1, 0.5)
    """
    rng = random.Random(seed)

    for u, v in sorted(graph.edges()):
        if latency_type == "linear":
            graph[u][v]["a"] = rng.uniform(0.5, 2.0)
            graph[u][v]["b"] = rng.uniform(0.5, 2.0)
        elif latency_type == "bpr":
            graph[u][v]["free_flow"] = rng.uniform(1.0, 5.0)
            # Scale capacity with demand so congestion is meaningful
            graph[u][v]["capacity"] = max(1.0, demand * rng.uniform(0.1, 0.5))
        else:
            raise ValueError(f"Unknown latency type: {latency_type}")


def compute_density(graph):
    """Compute edge density of a directed graph."""
    n = graph.number_of_nodes()
    if n <= 1:
        return 0.0
    max_edges = n * (n - 1)  # directed graph
    return graph.number_of_edges() / max_edges


# ─────────────────────────────────────────────────────────────────────
# CSV Output Helpers
# ─────────────────────────────────────────────────────────────────────

RESULTS_COLUMNS = [
    "experiment_id", "graph_type", "num_nodes", "num_edges", "density",
    "m_parameter", "p_parameter", "grid_rows", "grid_cols",
    "num_agents", "latency_type", "seed",
    "equilibrium_cost", "optimal_cost", "poa",
    "nash_iterations", "social_iterations",
    "runtime", "convergence_gap",
]

CONVERGENCE_COLUMNS = [
    "experiment_id", "solver_type", "iteration",
    "objective_value", "relative_gap", "step_size", "max_flow_change",
]

EDGE_FLOW_COLUMNS = [
    "experiment_id", "edge_source", "edge_target",
    "nash_flow", "social_flow",
    "edge_cost_nash", "edge_cost_social",
]


def ensure_results_dir(results_dir):
    """Create results directory if it doesn't exist."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)


def init_csv(filepath, columns):
    """Write CSV header if file doesn't exist."""
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()


def append_results_row(filepath, row_dict):
    """Append a single row to results.csv."""
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        writer.writerow(row_dict)


def append_convergence_rows(filepath, experiment_id, solver_type, convergence):
    """Append convergence data (one row per iteration) to convergence.csv."""
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CONVERGENCE_COLUMNS)
        for entry in convergence:
            writer.writerow({
                "experiment_id": experiment_id,
                "solver_type": solver_type,
                "iteration": entry["iteration"],
                "objective_value": entry["objective_value"],
                "relative_gap": entry["relative_gap"],
                "step_size": entry["step_size"],
                "max_flow_change": entry["max_flow_change"],
            })


def append_edge_flow_rows(filepath, experiment_id, graph, edges,
                          nash_flow, social_flow, latency_type):
    """Append edge-level flow data to edge_flows.csv."""
    from solver import edge_latency

    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EDGE_FLOW_COLUMNS)
        for u, v in edges:
            data = graph[u][v]
            nf = nash_flow.get((u, v), 0.0)
            sf = social_flow.get((u, v), 0.0)
            writer.writerow({
                "experiment_id": experiment_id,
                "edge_source": str(u),
                "edge_target": str(v),
                "nash_flow": nf,
                "social_flow": sf,
                "edge_cost_nash": edge_latency(data, nf, latency_type),
                "edge_cost_social": edge_latency(data, sf, latency_type),
            })
