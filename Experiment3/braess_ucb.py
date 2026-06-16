"""
UCB Bandit Agent on Braess's Network
=====================================

Applies the UCB1 multi-armed bandit approach from ``ucb_agent.py``
to the classical Braess network.

Braess's Network (4 nodes, 5 edges):
  Nodes: s(0), a(1), b(2), t(3)

  Edges and latency functions (scaled by 1/N):
    s → a :  l(x) = x / N   (congestion-dependent)
    a → t :  l(x) = 1        (constant)
    s → b :  l(x) = 1        (constant)
    b → t :  l(x) = x / N   (congestion-dependent)
    a → b :  l(x) = 0        (free shortcut — Braess edge)

  Three simple paths:
    Path 1:  s → a → t       latency = f_{sa}/N + 1
    Path 2:  s → b → t       latency = 1 + f_{bt}/N
    Path 3:  s → a → b → t   latency = f_{sa}/N + 0 + f_{bt}/N

Braess's Paradox:
  The shortcut edge (a → b) with zero cost makes Path 3 individually
  attractive, but when all agents use it, total cost *increases*.

Theoretical results:
  - Nash equilibrium : all N agents on Path 3
      per-agent cost = N/N + 0 + N/N = 2
      total Nash cost = 2N
  - Social optimum   : N/2 on Path 1, N/2 on Path 2 (shortcut unused)
      per-agent cost = (N/2)/N + 1 = 3/2
      total SO cost  = 3N/2
  - PoA = 2N / (3N/2) = 4/3

Reference:
  Braess, D. (1968). Über ein Paradoxon aus der Verkehrsplanung.
  Roughgarden, T. & Tardos, É. (2002). How bad is selfish routing?
  Auer, P., Cesa-Bianchi, N. & Fischer, P. (2002). UCB1.
"""

import os
import sys
import copy
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

# ── Import reusable components from the main experiment module ──
from ucb_agent import (
    UCBAgent,
    get_all_paths,
    compute_flow_dict,
    compute_path_latency,
    run_mixed_simulation,
    run_poa_vs_fraction_experiment,
    plot_poa_vs_fraction,
    plot_poa_convergence,
)


# ─────────────────────────────────────────────────────────────────────
# 1. Network Builder
# ─────────────────────────────────────────────────────────────────────

def build_braess_network(n_agents=50):
    """
    Build Braess's network as a ``nx.DiGraph`` with 4 nodes and 5 edges.

    Topology::

            a (1)
           / | \\
       x/N  |0  1
         /   |   \\
        s(0) |    t(3)
         \\   |   /
          1  |  x/N
           \\ | /
            b (2)

    Edges:
      (0,1) s→a : l(x) = x/N   [congestion-dependent]
      (1,3) a→t : l(x) = 1     [constant]
      (0,2) s→b : l(x) = 1     [constant]
      (2,3) b→t : l(x) = x/N   [congestion-dependent]
      (1,2) a→b : l(x) = 0     [free shortcut — Braess edge]

    Parameters
    ----------
    n_agents : int
        Number of agents (used to scale latency so PoA = 4/3).

    Returns
    -------
    G : networkx.DiGraph
        Directed graph with ``'a'``, ``'b'``, and ``'latency'``
        attributes on every edge.
    source : int
        Source node (0).
    sink : int
        Sink node (3).
    """
    N = float(n_agents)
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3])

    a_coeff = 1.0 / N

    # s(0) → a(1) :  l(x) = x / N
    G.add_edge(0, 1, a=a_coeff, b=0.0,
               latency=lambda x, _a=a_coeff: _a * x)

    # a(1) → t(3) :  l(x) = 1
    G.add_edge(1, 3, a=0.0, b=1.0,
               latency=lambda x: 1.0)

    # s(0) → b(2) :  l(x) = 1
    G.add_edge(0, 2, a=0.0, b=1.0,
               latency=lambda x: 1.0)

    # b(2) → t(3) :  l(x) = x / N
    G.add_edge(2, 3, a=a_coeff, b=0.0,
               latency=lambda x, _a=a_coeff: _a * x)

    # a(1) → b(2) :  l(x) = 0   (Braess shortcut)
    G.add_edge(1, 2, a=0.0, b=0.0,
               latency=lambda x: 0.0)

    return G, 0, 3


def analytical_braess_costs(n_agents):
    """
    Return the analytically known Nash and social-optimum costs for
    Braess's network with *n_agents* agents.

    Returns
    -------
    nash_cost : float
        Total cost at Nash equilibrium (all agents on s→a→b→t).
    opt_cost : float
        Total cost at social optimum (agents split between Paths 1 & 2).
    poa : float
        Price of Anarchy = nash_cost / opt_cost.
    """
    N = float(n_agents)
    nash_cost = 2.0 * N     # all on s→a→b→t: N × (1 + 0 + 1) = 2N
    opt_cost = 1.5 * N      # N/2 on each top/bottom path: N × 3/2
    poa = nash_cost / opt_cost  # 4/3
    return nash_cost, opt_cost, poa


# ─────────────────────────────────────────────────────────────────────
# 2. Main Block — Full Experiment
# ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    print("=" * 65)
    print("  UCB Bandit Simulation on BRAESS'S NETWORK")
    print("=" * 65)
    print()

    # ── Configuration ──
    N_AGENTS = 50
    N_STEPS  = 1000
    C_PARAM  = 1.41   # UCB exploration parameter (≈ √2)

    SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(SCRIPT_DIR, 'braess_results')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Build Braess network ──
    G, source, sink = build_braess_network(n_agents=N_AGENTS)
    paths = get_all_paths(G, source, sink)

    print(f"  Nodes      : {G.number_of_nodes()}")
    print(f"  Edges      : {G.number_of_edges()}")
    print(f"  Paths      : {len(paths)}")
    for i, p in enumerate(paths):
        print(f"    Path {i+1}  : {p}")
    print()

    # ── 2. Analytical costs ──
    nash_cost_th, opt_cost_th, poa_th = analytical_braess_costs(N_AGENTS)
    print(f"  Analytical Nash cost    : {nash_cost_th:.2f}")
    print(f"  Analytical SO cost      : {opt_cost_th:.2f}")
    print(f"  Analytical PoA          : {poa_th:.6f}")
    print()

    # ── 3. Compute social-optimum cost via Frank-Wolfe ──
    EXP1_DIR = os.path.join(SCRIPT_DIR, '..', 'Experiment1')
    sys.path.insert(0, os.path.abspath(EXP1_DIR))
    from solver import compute_social_optimum, total_system_cost

    G_fw = copy.deepcopy(G)
    social_flow, social_iters, _ = compute_social_optimum(
        G_fw, source, sink, demand=N_AGENTS,
        latency_type="linear", max_iter=500, tol=1e-8,
    )
    edges_fw = sorted(G_fw.edges())
    opt_cost = total_system_cost(G_fw, social_flow, edges_fw, "linear")
    print(f"  Frank-Wolfe SO cost     : {opt_cost:.4f}  "
          f"({social_iters} iterations)")
    print(f"  (Analytical check       : {opt_cost_th:.4f})")
    print()

    # ── 4. Mixed simulation (50% bandit) ──
    print("-" * 65)
    print("  Mixed Simulation  (50% UCB bandit, 50% selfish)")
    print("-" * 65)
    sim_result = run_mixed_simulation(
        G, source, sink,
        n_agents=N_AGENTS,
        pct_bandit=0.5,
        c=C_PARAM,
        n_steps=N_STEPS,
        opt_cost=opt_cost,
    )
    valid_poas = [p for p in sim_result['poa_history'] if p is not None]
    if valid_poas:
        print(f"  Final PoA (last 50 steps): {np.mean(valid_poas[-50:]):.4f}")
    print()

    # ── 5. PoA vs fraction sweep ──
    print("-" * 65)
    print("  PoA vs Fraction of UCB Agents")
    print("-" * 65)
    fraction_results = run_poa_vs_fraction_experiment(
        G, source, sink,
        n_agents=N_AGENTS,
        c=C_PARAM,
        n_steps=N_STEPS,
        n_repeats=10,
        opt_cost=opt_cost,
    )

    # ── 6. Summary table ──
    print()
    print("-" * 50)
    print(f"{'Fraction':>10s}  {'Mean PoA':>10s}  {'Std PoA':>10s}")
    print("-" * 50)
    for frac in sorted(fraction_results.keys()):
        r = fraction_results[frac]
        print(f"  {frac:>7.0%}    {r['mean_poa']:>10.4f}  {r['std_poa']:>10.4f}")
    print("-" * 50)

    # ── 7. Generate plots ──
    print("\nGenerating plots ...")

    # Plot 1: PoA vs fraction
    plot_poa_vs_fraction(
        fraction_results,
        save_path=os.path.join(RESULTS_DIR, 'braess_poa_vs_rl_fraction.png'),
    )

    # Plot 2: Convergence curves for representative fractions
    convergence_dict = {}
    for frac_label, frac_val in tqdm(
        [('0% bandit', 0.0),
         ('25% bandit', 0.25),
         ('50% bandit', 0.5),
         ('100% bandit', 1.0)],
        desc="Convergence runs",
    ):
        conv_sim = run_mixed_simulation(
            G, source, sink,
            n_agents=N_AGENTS,
            pct_bandit=frac_val,
            c=C_PARAM,
            n_steps=N_STEPS,
            opt_cost=opt_cost,
        )
        convergence_dict[frac_label] = conv_sim['poa_history']

    plot_poa_convergence(
        convergence_dict,
        save_path=os.path.join(RESULTS_DIR, 'braess_poa_convergence.png'),
    )

    print(f"\nDone.  Results saved to: {RESULTS_DIR}")
