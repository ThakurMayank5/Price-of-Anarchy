"""
Traffic Assignment Experiment Runner
=====================================

Runs Frank-Wolfe traffic assignment experiments across three graph types:
  1. Grid graphs       (10x10, 30x30, 50x50, 100x100)
  2. Erdos-Renyi       (20, 50, 100 nodes — sparse & dense)
  3. Scale-Free (BA)   (20, 50, 100 nodes — sparse & dense)

For each configuration:
  - Generates the graph
  - Assigns latency functions (linear, BPR)
  - Defines OD demand (100 to 10000 agents)
  - Computes Wardrop equilibrium (Frank-Wolfe)
  - Computes social optimum (Frank-Wolfe with marginal costs)
  - Calculates Price of Anarchy
  - Records convergence metrics
  - Saves everything to CSV
"""

import os
import sys
import time
import copy

# Ensure imports work from this directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from solver import (
    compute_wardrop_equilibrium,
    compute_social_optimum,
    compute_poa,
)
from utils import (
    generate_grid_graph,
    generate_erdos_renyi_graph,
    generate_scale_free_graph,
    assign_latency,
    compute_density,
    ensure_results_dir,
    init_csv,
    append_results_row,
    append_convergence_rows,
    append_edge_flow_rows,
    RESULTS_COLUMNS,
    CONVERGENCE_COLUMNS,
    EDGE_FLOW_COLUMNS,
)
from plots import generate_all_plots


# ─────────────────────────────────────────────────────────────────────
# Experiment Configuration
# ─────────────────────────────────────────────────────────────────────

AGENT_COUNTS = [100, 500, 1000, 5000, 10000]

LATENCY_TYPES = ["linear", "bpr"]

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Frank-Wolfe parameters
MAX_ITER = 300
TOLERANCE = 1e-6
BASE_SEED = 42


# ─────────────────────────────────────────────────────────────────────
# Core Experiment Runner
# ─────────────────────────────────────────────────────────────────────

def run_single_experiment(experiment_id, graph, source, target,
                          num_agents, latency_type, seed,
                          graph_type, m_param=None, p_param=None,
                          grid_rows=None, grid_cols=None):
    """
    Run one complete experiment:
      1. Assign latency functions to edges
      2. Compute Wardrop equilibrium
      3. Compute social optimum
      4. Calculate PoA
      5. Save results, convergence, and edge flows to CSV

    Returns the results dict for this experiment.
    """
    results_csv = os.path.join(RESULTS_DIR, "results.csv")
    conv_csv = os.path.join(RESULTS_DIR, "convergence.csv")
    edge_csv = os.path.join(RESULTS_DIR, "edge_flows.csv")

    # ── Step 1: Assign latency ──
    assign_latency(graph, latency_type, demand=num_agents, seed=seed)

    edges = sorted(graph.edges())
    start_time = time.time()

    # ── Step 2: Wardrop (Nash) equilibrium ──
    # Use a copy so Nash and Social don't interfere
    graph_nash = copy.deepcopy(graph)
    nash_flow, nash_iters, nash_conv = compute_wardrop_equilibrium(
        graph_nash, source, target, demand=num_agents,
        latency_type=latency_type, max_iter=MAX_ITER, tol=TOLERANCE,
    )

    # ── Step 3: Social optimum ──
    graph_social = copy.deepcopy(graph)
    assign_latency(graph_social, latency_type, demand=num_agents, seed=seed)
    social_flow, social_iters, social_conv = compute_social_optimum(
        graph_social, source, target, demand=num_agents,
        latency_type=latency_type, max_iter=MAX_ITER, tol=TOLERANCE,
    )

    runtime = time.time() - start_time

    # ── Step 4: Price of Anarchy ──
    # Re-assign latency to a clean graph for fair cost comparison
    graph_eval = copy.deepcopy(graph)
    assign_latency(graph_eval, latency_type, demand=num_agents, seed=seed)
    edges_eval = sorted(graph_eval.edges())
    poa, nash_cost, social_cost = compute_poa(
        graph_eval, nash_flow, social_flow, edges_eval, latency_type
    )

    # Final convergence gap
    final_gap = nash_conv[-1]["relative_gap"] if nash_conv else 0.0

    # ── Step 5: Save to CSV ──
    row = {
        "experiment_id": experiment_id,
        "graph_type": graph_type,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "density": round(compute_density(graph), 6),
        "m_parameter": m_param,
        "p_parameter": p_param,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "num_agents": num_agents,
        "latency_type": latency_type,
        "seed": seed,
        "equilibrium_cost": round(nash_cost, 6),
        "optimal_cost": round(social_cost, 6),
        "poa": round(poa, 6),
        "nash_iterations": nash_iters,
        "social_iterations": social_iters,
        "runtime": round(runtime, 4),
        "convergence_gap": round(final_gap, 10),
    }
    append_results_row(results_csv, row)

    # ── Step 6: Save convergence data ──
    append_convergence_rows(conv_csv, experiment_id, "nash", nash_conv)
    append_convergence_rows(conv_csv, experiment_id, "social", social_conv)

    # ── Step 7: Save edge flows (skip for very large graphs to keep file size sane) ──
    if graph.number_of_nodes() <= 2500:
        append_edge_flow_rows(
            edge_csv, experiment_id, graph_eval, edges_eval,
            nash_flow, social_flow, latency_type,
        )

    return row


def print_result(row):
    """Pretty-print a single experiment result."""
    print(
        f"  {row['experiment_id']:45s} | "
        f"Nodes={row['num_nodes']:>5d} | "
        f"PoA={row['poa']:>8.4f} | "
        f"Nash={row['equilibrium_cost']:>12.2f} | "
        f"Social={row['optimal_cost']:>12.2f} | "
        f"Time={row['runtime']:>6.2f}s"
    )


# ─────────────────────────────────────────────────────────────────────
# Grid Graph Experiments
# ─────────────────────────────────────────────────────────────────────

def run_grid_graph_experiment():
    """
    Run experiments on Grid graphs.
    Sizes: 10x10, 30x30, 50x50, 100x100
    Source: top-left (0,0), Target: bottom-right (rows-1, cols-1)
    """
    print("\n" + "=" * 70)
    print("GRID GRAPH EXPERIMENTS")
    print("=" * 70)

    grid_sizes = [10, 30, 50, 100]

    for dim in grid_sizes:
        graph = generate_grid_graph(dim, dim)
        source = (0, 0)
        target = (dim - 1, dim - 1)

        print(f"\n--- Grid {dim}x{dim}  "
              f"(nodes={graph.number_of_nodes()}, "
              f"edges={graph.number_of_edges()}) ---")

        for latency in LATENCY_TYPES:
            for agents in AGENT_COUNTS:
                exp_id = f"grid_{dim}x{dim}_{latency}_{agents}"
                seed = BASE_SEED + dim + agents

                try:
                    row = run_single_experiment(
                        experiment_id=exp_id,
                        graph=copy.deepcopy(graph),
                        source=source, target=target,
                        num_agents=agents, latency_type=latency, seed=seed,
                        graph_type="grid",
                        grid_rows=dim, grid_cols=dim,
                    )
                    print_result(row)
                except Exception as e:
                    print(f"  ERROR {exp_id}: {e}")


# ─────────────────────────────────────────────────────────────────────
# Erdos-Renyi Random Graph Experiments
# ─────────────────────────────────────────────────────────────────────

def run_random_graph_experiment():
    """
    Run experiments on Erdos-Renyi random graphs.
    Nodes: 20, 50, 100
    Configurations: sparse (low p) and dense (high p)
    """
    print("\n" + "=" * 70)
    print("ERDOS-RENYI RANDOM GRAPH EXPERIMENTS")
    print("=" * 70)

    er_configs = [
        # (nodes, p_sparse, p_dense)
        (20,  0.15, 0.40),
        (50,  0.08, 0.30),
        (100, 0.05, 0.20),
    ]

    for n, p_sparse, p_dense in er_configs:
        for density_label, p in [("sparse", p_sparse), ("dense", p_dense)]:
            try:
                graph, source, target = generate_erdos_renyi_graph(
                    n, p, seed=BASE_SEED + n
                )
            except ValueError as e:
                print(f"  SKIP ER n={n} p={p}: {e}")
                continue

            print(f"\n--- ER n={n} p={p} ({density_label})  "
                  f"(nodes={graph.number_of_nodes()}, "
                  f"edges={graph.number_of_edges()}) ---")

            for latency in LATENCY_TYPES:
                for agents in AGENT_COUNTS:
                    exp_id = f"er_{n}_{density_label}_{latency}_{agents}"
                    seed = BASE_SEED + n + agents + int(p * 100)

                    try:
                        row = run_single_experiment(
                            experiment_id=exp_id,
                            graph=copy.deepcopy(graph),
                            source=source, target=target,
                            num_agents=agents, latency_type=latency,
                            seed=seed, graph_type="erdos_renyi",
                            p_param=p,
                        )
                        print_result(row)
                    except Exception as e:
                        print(f"  ERROR {exp_id}: {e}")


# ─────────────────────────────────────────────────────────────────────
# Scale-Free Graph Experiments
# ─────────────────────────────────────────────────────────────────────

def run_scale_free_graph_experiment():
    """
    Run experiments on Barabasi-Albert scale-free graphs.
    Nodes: 20, 50, 100
    Configurations: sparse (m=2) and dense (m=5)
    """
    print("\n" + "=" * 70)
    print("SCALE-FREE GRAPH EXPERIMENTS")
    print("=" * 70)

    sf_configs = [
        # (nodes, m_sparse, m_dense)
        (20,  2, 5),
        (50,  2, 5),
        (100, 2, 5),
    ]

    for n, m_sparse, m_dense in sf_configs:
        for density_label, m in [("sparse", m_sparse), ("dense", m_dense)]:
            graph = generate_scale_free_graph(n, m, seed=BASE_SEED + n)

            # Pick source=0, target=farthest node
            from utils import pick_source_target_general
            source, target = pick_source_target_general(graph)
            if source is None:
                print(f"  SKIP SF n={n} m={m}: no valid path")
                continue

            print(f"\n--- Scale-Free n={n} m={m} ({density_label})  "
                  f"(nodes={graph.number_of_nodes()}, "
                  f"edges={graph.number_of_edges()}) ---")

            for latency in LATENCY_TYPES:
                for agents in AGENT_COUNTS:
                    exp_id = f"sf_{n}_m{m}_{density_label}_{latency}_{agents}"
                    seed = BASE_SEED + n + agents + m

                    try:
                        row = run_single_experiment(
                            experiment_id=exp_id,
                            graph=copy.deepcopy(graph),
                            source=source, target=target,
                            num_agents=agents, latency_type=latency,
                            seed=seed, graph_type="scale_free",
                            m_param=m,
                        )
                        print_result(row)
                    except Exception as e:
                        print(f"  ERROR {exp_id}: {e}")


# ─────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Traffic Assignment Experiment")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Agent counts: {AGENT_COUNTS}")
    print(f"Latency types: {LATENCY_TYPES}")
    print(f"Max FW iterations: {MAX_ITER}, Tolerance: {TOLERANCE}")

    # Initialize output directory and CSV headers
    ensure_results_dir(RESULTS_DIR)
    results_csv = os.path.join(RESULTS_DIR, "results.csv")
    conv_csv = os.path.join(RESULTS_DIR, "convergence.csv")
    edge_csv = os.path.join(RESULTS_DIR, "edge_flows.csv")
    init_csv(results_csv, RESULTS_COLUMNS)
    init_csv(conv_csv, CONVERGENCE_COLUMNS)
    init_csv(edge_csv, EDGE_FLOW_COLUMNS)

    total_start = time.time()

    # Run all three graph types
    run_grid_graph_experiment()
    run_random_graph_experiment()
    run_scale_free_graph_experiment()

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETE — Total time: {total_time:.1f}s")
    print(f"{'=' * 70}")

    # Generate all plots from the CSV files
    generate_all_plots(RESULTS_DIR)
