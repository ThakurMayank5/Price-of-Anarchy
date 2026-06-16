"""
Thompson Sampling Bandit Agent for Congestion Game Simulation
==============================================================

Implements a multi-armed bandit approach to route selection in congestion
games.  Each agent treats the available source->sink paths as "arms" and
uses Thompson Sampling to balance exploration with exploitation.

For a *minimisation* problem with continuous (latency) rewards, the agent
models the latency of each arm as a Normal distribution with unknown mean
and variance, using a Normal-Inverse-Gamma (NIG) conjugate prior.

At each step:
  1.  For every arm *i*, sample a latency value from the posterior
      predictive distribution (a Student-t):
          θ_i ~ Student-t(2·α_i, μ_i, β_i / (α_i · κ_i))
  2.  Select the arm with the *lowest* sampled θ_i  (minimisation).
  3.  After observing the actual latency, update the posterior parameters.

NIG update rules (Normal-Inverse-Gamma conjugate model):
  κ_n = κ_0 + n
  μ_n = (κ_0·μ_0 + n·x̄) / κ_n
  α_n = α_0 + n/2
  β_n = β_0 + ½·Σ(x_i − x̄)² + (κ_0·n·(x̄ − μ_0)²) / (2·κ_n)

For online (incremental) updates we use the simplified one-observation
form:
  κ' = κ + 1
  μ' = (κ·μ + x) / κ'
  α' = α + 0.5
  β' = β + κ·(x − μ)² / (2·κ')

Key components:
  1. ThompsonSamplingAgent    – bandit agent with NIG posterior updates
  2. Helper functions         – path enumeration, latency computation
  3. Mixed simulation runner  – bandit + selfish agents on shared network
  4. Experiment runner        – PoA vs fraction of RL agents
  5. Visualization            – convergence curves & PoA-vs-fraction plots

Theoretical reference:
  Price of Anarchy <= 4/3 ~ 1.333 for linear latency functions
  (Roughgarden & Tardos, 2002)

  Thompson Sampling:
  Thompson, W. R. (1933). On the likelihood that one unknown probability
  exceeds another in view of the evidence of two samples.
  Biometrika, 25(3/4), 285-294.

  Agrawal, S. & Goyal, N. (2012). Analysis of Thompson Sampling for the
  Multi-armed Bandit Problem. COLT 2012.
"""

import os
import sys
import random
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────
# 1. Thompson Sampling Bandit Agent
# ─────────────────────────────────────────────────────────────────────

class ThompsonSamplingAgent:
    """
    A Thompson Sampling multi-armed bandit agent for path selection in a
    congestion game (latency minimisation).

    Each available path is treated as an arm.  The agent maintains a
    Normal-Inverse-Gamma (NIG) posterior over the unknown mean and
    variance of each arm's latency distribution.

    At each step the agent:
      1. Samples a latency value from the posterior predictive (Student-t)
         for every arm.
      2. Chooses the arm with the *lowest* sampled latency.
      3. After observing the realised latency, updates the NIG posterior.

    Parameters
    ----------
    agent_id : int or str
        Unique identifier for this agent.
    paths : list[list]
        Available paths from source to sink.  Each path is a list of
        node IDs, e.g. [[0, 3, 7, 14], [0, 5, 14]].
    mu_0 : float, optional
        Prior mean of the latency (default 0.0).
    kappa_0 : float, optional
        Prior strength (pseudo-observations for the mean, default 1.0).
    alpha_0 : float, optional
        Prior shape parameter for Inverse-Gamma (default 1.0).
        Must be > 0.  Smaller → wider prior → more exploration.
    beta_0 : float, optional
        Prior scale parameter for Inverse-Gamma (default 1.0).

    Attributes
    ----------
    counts : numpy.ndarray
        Number of times each path has been chosen.
    mu : numpy.ndarray
        Posterior mean of the latency for each arm.
    kappa : numpy.ndarray
        Posterior strength for each arm.
    alpha : numpy.ndarray
        Posterior shape for each arm.
    beta : numpy.ndarray
        Posterior scale for each arm.
    current_path_idx : int
        Index of the most recently chosen path.
    """

    def __init__(self, agent_id, paths,
                 mu_0=0.0, kappa_0=1.0, alpha_0=1.0, beta_0=1.0):
        if not paths:
            raise ValueError("Agent must have at least one available path.")
        self.agent_id = agent_id
        self.paths = paths
        self.n_paths = len(paths)

        # NIG prior hyper-parameters (stored per-arm)
        self.mu = np.full(self.n_paths, mu_0, dtype=np.float64)
        self.kappa = np.full(self.n_paths, kappa_0, dtype=np.float64)
        self.alpha = np.full(self.n_paths, alpha_0, dtype=np.float64)
        self.beta = np.full(self.n_paths, beta_0, dtype=np.float64)

        self.counts = np.zeros(self.n_paths, dtype=np.float64)

        # Initialise with a random path choice
        self.current_path_idx = random.randint(0, self.n_paths - 1)

    def choose_path(self):
        """
        Select a path using Thompson Sampling (minimisation).

        For each arm *i* the agent samples from the posterior predictive
        Student-t distribution:

            θ_i ~ Student-t(df=2·α_i,  loc=μ_i,
                            scale=sqrt(β_i / (α_i · κ_i)))

        and selects the arm with the *lowest* sampled θ_i.

        Returns
        -------
        list
            The chosen path (list of node IDs).
        """
        # If only one path exists, always return it — no decision to make
        if self.n_paths == 1:
            self.current_path_idx = 0
            return self.paths[0]

        # Sample from posterior predictive (Student-t) for each arm
        sampled = np.empty(self.n_paths, dtype=np.float64)
        for i in range(self.n_paths):
            df = 2.0 * self.alpha[i]
            loc = self.mu[i]
            # scale of the Student-t predictive
            scale = math.sqrt(self.beta[i] / (self.alpha[i] * self.kappa[i]))
            # np.random returns a standard t; shift and scale manually
            sampled[i] = loc + scale * np.random.standard_t(df)

        # Pick the arm with the lowest sampled latency (minimisation)
        min_val = np.min(sampled)
        best_indices = np.where(sampled == min_val)[0]
        self.current_path_idx = int(random.choice(best_indices))

        return self.paths[self.current_path_idx]

    def update(self, latency_experienced):
        """
        Update the NIG posterior for the most recently chosen path
        after observing a single latency value.

        Incremental NIG update (one observation x):
            κ' = κ + 1
            μ' = (κ·μ + x) / κ'
            α' = α + 0.5
            β' = β + κ·(x − μ)² / (2·κ')

        Parameters
        ----------
        latency_experienced : float
            The actual latency the agent observed after all agents
            committed to their paths for this step.
        """
        idx = self.current_path_idx
        x = latency_experienced

        kappa_old = self.kappa[idx]
        mu_old = self.mu[idx]

        kappa_new = kappa_old + 1.0
        mu_new = (kappa_old * mu_old + x) / kappa_new
        alpha_new = self.alpha[idx] + 0.5
        beta_new = self.beta[idx] + kappa_old * (x - mu_old) ** 2 / (2.0 * kappa_new)

        self.kappa[idx] = kappa_new
        self.mu[idx] = mu_new
        self.alpha[idx] = alpha_new
        self.beta[idx] = beta_new

        self.counts[idx] += 1

    def get_current_path(self):
        """
        Return the path the agent most recently chose.

        Returns
        -------
        list
            Path as a list of node IDs.
        """
        return self.paths[self.current_path_idx]


# ─────────────────────────────────────────────────────────────────────
# 2. Helper Functions
# ─────────────────────────────────────────────────────────────────────

def get_all_paths(G, source, sink, cutoff=6):
    """
    Enumerate all simple paths from *source* to *sink* in digraph *G*,
    up to a maximum path length of *cutoff* edges.

    Uses ``nx.all_simple_paths`` internally with a cutoff to prevent
    combinatorial explosion on large or dense graphs.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed graph.
    source : node
        Source node.
    sink : node
        Destination node.
    cutoff : int, optional
        Maximum number of edges in any returned path (default 6).

    Returns
    -------
    list[list]
        List of paths, each a list of node IDs.

    Raises
    ------
    ValueError
        If no paths exist between source and sink.
    """
    paths = list(nx.all_simple_paths(G, source, sink, cutoff=cutoff))
    if not paths:
        raise ValueError(
            "No paths found between source and sink -- try a denser graph"
        )
    return paths


def compute_flow_dict(chosen_paths):
    """
    Given the paths chosen by every agent in a single step, compute
    the flow on each edge (i.e. the number of agents traversing it).

    Parameters
    ----------
    chosen_paths : list[list]
        One path per agent.  Each path is a list of node IDs.

    Returns
    -------
    dict
        Mapping ``(u, v) -> int`` with the count of agents on edge (u, v).
    """
    flow = defaultdict(int)
    for path in chosen_paths:
        for u, v in zip(path[:-1], path[1:]):
            flow[(u, v)] += 1
    return dict(flow)


def compute_path_latency(G, path, flow_dict):
    """
    Compute the total latency experienced by an agent traversing *path*,
    given the current edge flows.

    Each edge must carry a ``'latency'`` attribute that is a callable
    ``latency_fn(flow) -> float``.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed graph with ``G[u][v]['latency']`` callables.
    path : list
        Sequence of node IDs defining the path.
    flow_dict : dict
        Mapping ``(u, v) -> int`` with the number of agents on each edge.

    Returns
    -------
    float
        Sum of edge latencies along the path.
    """
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge_flow = flow_dict.get((u, v), 0)
        latency_fn = G[u][v]['latency']
        total += latency_fn(edge_flow)
    return total


# ─────────────────────────────────────────────────────────────────────
# 3. Mixed Simulation Runner
# ─────────────────────────────────────────────────────────────────────

def run_mixed_simulation(G, source, sink, n_agents=50, pct_bandit=0.5,
                         mu_0=0.0, kappa_0=1.0, alpha_0=1.0, beta_0=1.0,
                         n_steps=300, opt_cost=None):
    """
    Run a mixed simulation where a fraction of agents use Thompson
    Sampling bandit learning and the rest are selfish (greedy
    best-response).

    Selfish agents recalculate the lowest-latency path at every step
    based on the *current* flow (no memory).  Bandit agents use their
    internal posteriors and update after observing realised latency.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed graph with ``G[u][v]['latency']`` callables on every edge.
    source : node
        Source node.
    sink : node
        Destination / sink node.
    n_agents : int, optional
        Total number of agents (default 50).
    pct_bandit : float, optional
        Fraction of agents that are Thompson Sampling bandits (default 0.5).
    mu_0 : float, optional
        Prior mean for the NIG prior (default 0.0).
    kappa_0 : float, optional
        Prior strength (default 1.0).
    alpha_0 : float, optional
        Prior shape (default 1.0).
    beta_0 : float, optional
        Prior scale (default 1.0).
    n_steps : int, optional
        Number of simulation steps (default 300).
    opt_cost : float or None, optional
        Social-optimum cost.  If provided the PoA is recorded each step.

    Returns
    -------
    dict
        ``'poa_history'``       – list of PoA values (one per step);
                                   ``None`` entries if *opt_cost* is not given.
        ``'final_flow_dict'``   – edge flow counts at the last step.
        ``'agent_path_counts'`` – for each bandit agent, a numpy array of
                                   per-path selection counts.
    """
    # ── Enumerate paths ──
    all_paths = get_all_paths(G, source, sink)

    # ── Create agents ──
    n_bandit = int(round(n_agents * pct_bandit))
    n_selfish = n_agents - n_bandit

    bandit_agents = [
        ThompsonSamplingAgent(
            agent_id=i, paths=all_paths,
            mu_0=mu_0, kappa_0=kappa_0,
            alpha_0=alpha_0, beta_0=beta_0,
        )
        for i in range(n_bandit)
    ]

    poa_history = []
    flow_dict = {}

    for step in tqdm(range(n_steps), desc="Simulating", leave=False):
        chosen_paths = []

        # ── 1. Bandit agents choose paths ──
        for agent in bandit_agents:
            chosen_paths.append(agent.choose_path())

        # ── 2. Selfish agents choose lowest-latency path on current flows ──
        # First pass: compute preliminary flows from bandit choices so that
        # selfish agents can react to the current state.
        preliminary_flow = compute_flow_dict(chosen_paths)

        for _ in range(n_selfish):
            # Evaluate each path under current preliminary flows
            best_path = None
            best_latency = float('inf')
            for path in all_paths:
                lat = compute_path_latency(G, path, preliminary_flow)
                if lat < best_latency:
                    best_latency = lat
                    best_path = path
            chosen_paths.append(best_path)
            # Update preliminary flow with this selfish choice so that
            # subsequent selfish agents see the updated congestion
            for u, v in zip(best_path[:-1], best_path[1:]):
                preliminary_flow[(u, v)] = preliminary_flow.get((u, v), 0) + 1

        # ── 3. Compute final flow dict ──
        flow_dict = compute_flow_dict(chosen_paths)

        # ── 4. Compute Nash social cost = sum of path latencies ──
        nash_cost = 0.0
        for path in chosen_paths:
            nash_cost += compute_path_latency(G, path, flow_dict)

        # ── 5. Record PoA ──
        if opt_cost is not None and opt_cost > 1e-15:
            poa_history.append(nash_cost / opt_cost)
        else:
            poa_history.append(None)

        # ── 6. Update bandit agents with experienced latency ──
        for agent in bandit_agents:
            latency = compute_path_latency(G, agent.get_current_path(),
                                           flow_dict)
            agent.update(latency)

    # ── Collect bandit path-selection statistics ──
    agent_path_counts = {
        agent.agent_id: agent.counts.copy() for agent in bandit_agents
    }

    return {
        'poa_history': poa_history,
        'final_flow_dict': flow_dict,
        'agent_path_counts': agent_path_counts,
    }


# ─────────────────────────────────────────────────────────────────────
# 4. PoA vs RL-Fraction Experiment
# ─────────────────────────────────────────────────────────────────────

def run_poa_vs_fraction_experiment(G, source, sink, n_agents=50,
                                   fractions=None,
                                   mu_0=0.0, kappa_0=1.0,
                                   alpha_0=1.0, beta_0=1.0,
                                   n_steps=300, n_repeats=10,
                                   opt_cost=None):
    """
    Sweep the fraction of Thompson Sampling agents from 0% to 100% and
    record the resulting mean final PoA.

    For each fraction the simulation is repeated *n_repeats* times;
    the "final PoA" of a single run is the average of the last 50
    steps of its ``poa_history``.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed graph with ``G[u][v]['latency']`` callables.
    source : node
        Source node.
    sink : node
        Destination node.
    n_agents : int, optional
        Total number of agents (default 50).
    fractions : list[float] or None, optional
        Fractions to test (default ``[0.0, 0.1, 0.25, 0.5, 0.75, 1.0]``).
    mu_0, kappa_0, alpha_0, beta_0 : float, optional
        NIG prior hyper-parameters passed to each agent.
    n_steps : int, optional
        Number of simulation steps per run (default 300).
    n_repeats : int, optional
        Number of independent repetitions per fraction (default 10).
    opt_cost : float
        Pre-computed social-optimum cost.  Must be supplied explicitly
        via Frank-Wolfe solver.  Do NOT pass None.

    Returns
    -------
    dict
        Mapping ``fraction -> {'mean_poa': float, 'std_poa': float}``.

    Raises
    ------
    ValueError
        If *opt_cost* is None.
    """
    if opt_cost is None:
        raise ValueError(
            "opt_cost must be supplied explicitly via Frank-Wolfe solver. "
            "Do not pass None — this prevents silent PoA < 1.0 errors."
        )

    if fractions is None:
        fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    results = {}
    for frac in tqdm(fractions, desc="Fraction sweep"):
        final_poas = []
        for rep in tqdm(range(n_repeats), desc=f"  Repeats frac={frac:.0%}", leave=False):
            sim = run_mixed_simulation(
                G, source, sink,
                n_agents=n_agents,
                pct_bandit=frac,
                mu_0=mu_0, kappa_0=kappa_0,
                alpha_0=alpha_0, beta_0=beta_0,
                n_steps=n_steps,
                opt_cost=opt_cost,
            )
            # Final PoA = mean of last 50 steps
            tail = [p for p in sim['poa_history'][-50:] if p is not None]
            if tail:
                final_poas.append(float(np.mean(tail)))

        mean_poa = float(np.mean(final_poas)) if final_poas else float('nan')
        std_poa = float(np.std(final_poas)) if final_poas else float('nan')
        results[frac] = {'mean_poa': mean_poa, 'std_poa': std_poa}
        print(f"  Fraction={frac:.2f}  |  Mean PoA={mean_poa:.4f}  "
              f"±  {std_poa:.4f}  ({len(final_poas)} runs)")

    return results





# ─────────────────────────────────────────────────────────────────────
# 5. Visualization Functions
# ─────────────────────────────────────────────────────────────────────

# -- Styling constants ------------------------------------------------
_COLORS = {
    'primary':     '#6C5CE7',
    'secondary':   '#00CEC9',
    'accent':      '#FD79A8',
    'dark_bg':     '#1A1A2E',
    'panel_bg':    '#16213E',
    'grid_color':  '#2A2A4A',
    'text':        '#E0E0E0',
    'bound_line':  '#FF6B6B',
    'opt_line':    '#55EFC4',
}

_PALETTE = ['#6C5CE7', '#00CEC9', '#FD79A8', '#FDCB6E',
            '#74B9FF', '#A29BFE', '#FF7675', '#55EFC4']


def _apply_dark_style():
    """Apply a modern dark theme to matplotlib plots."""
    plt.rcParams.update({
        'figure.facecolor':   _COLORS['dark_bg'],
        'axes.facecolor':     _COLORS['panel_bg'],
        'axes.edgecolor':     _COLORS['grid_color'],
        'axes.labelcolor':    _COLORS['text'],
        'axes.titlecolor':    _COLORS['text'],
        'xtick.color':        _COLORS['text'],
        'ytick.color':        _COLORS['text'],
        'text.color':         _COLORS['text'],
        'axes.grid':          True,
        'grid.color':         _COLORS['grid_color'],
        'grid.alpha':         0.4,
        'font.size':          12,
        'axes.titlesize':     14,
        'axes.labelsize':     12,
        'legend.fontsize':    10,
        'legend.facecolor':   _COLORS['panel_bg'],
        'legend.edgecolor':   _COLORS['grid_color'],
        'figure.dpi':         150,
        'savefig.facecolor':  _COLORS['dark_bg'],
    })


def plot_poa_vs_fraction(results_dict, save_path='poa_vs_rl_fraction.png'):
    """
    Plot the mean empirical PoA as a function of the fraction of
    Thompson Sampling agents, with ±1 std shading.

    Parameters
    ----------
    results_dict : dict
        Output of ``run_poa_vs_fraction_experiment``.
        Mapping ``fraction -> {'mean_poa', 'std_poa'}``.
    save_path : str, optional
        File path for the saved figure (default ``'poa_vs_rl_fraction.png'``).
    """
    _apply_dark_style()

    fractions = sorted(results_dict.keys())
    means = [results_dict[f]['mean_poa'] for f in fractions]
    stds = [results_dict[f]['std_poa'] for f in fractions]

    means = np.array(means)
    stds = np.array(stds)
    fracs = np.array(fractions) * 100  # convert to percentage

    fig, ax = plt.subplots(figsize=(10, 6))

    # Shaded std region
    ax.fill_between(fracs, means - stds, means + stds,
                    color=_COLORS['primary'], alpha=0.25, linewidth=0)

    # Main curve
    ax.plot(fracs, means, 'o-', color=_COLORS['primary'],
            linewidth=2.5, markersize=8, markeredgecolor='white',
            markeredgewidth=1.5, label='Mean PoA', zorder=5)

    # Reference lines
    ax.axhline(y=4/3, color=_COLORS['bound_line'], linestyle='--',
               linewidth=1.5, alpha=0.85, label='Theoretical bound (4/3)')
    ax.axhline(y=1.0, color=_COLORS['opt_line'], linestyle='--',
               linewidth=1.5, alpha=0.85, label='Social optimum (1.0)')

    ax.set_xlabel('Fraction of Thompson Sampling Agents (%)')
    ax.set_ylabel('Mean Empirical PoA')
    ax.set_title('PoA vs Fraction of Thompson Sampling Agents')
    ax.legend(loc='best')
    ax.set_xlim(-2, 102)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_poa_convergence(poa_history_dict, save_path='poa_convergence.png'):
    """
    Plot PoA convergence curves for multiple simulation runs on the
    same axes.

    Parameters
    ----------
    poa_history_dict : dict
        Mapping ``label -> poa_history_list``.  Each list contains one
        PoA value per simulation step.
    save_path : str, optional
        File path for the saved figure (default ``'poa_convergence.png'``).
    """
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (label, history) in enumerate(poa_history_dict.items()):
        color = _PALETTE[idx % len(_PALETTE)]
        valid = [h if h is not None else float('nan') for h in history]
        ax.plot(valid, linewidth=1.5, alpha=0.85, color=color, label=label)

    # Reference lines
    ax.axhline(y=4/3, color=_COLORS['bound_line'], linestyle='--',
               linewidth=1.5, alpha=0.85, label='Theoretical bound (4/3)')
    ax.axhline(y=1.0, color=_COLORS['opt_line'], linestyle='--',
               linewidth=1.5, alpha=0.85, label='Social optimum (1.0)')

    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Empirical PoA')
    ax.set_title('PoA Convergence Over Time (Thompson Sampling Agents)')
    ax.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────
# 6. Main Block — Quick End-to-End Test
# ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    print("Starting simulation ...")
    print("Progress bars will appear below.\n")

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -- Import Frank-Wolfe solver from Experiment1 --
    EXP1_DIR = os.path.join(SCRIPT_DIR, '..', 'Experiment1')
    sys.path.insert(0, os.path.abspath(EXP1_DIR))
    from solver import (
        compute_social_optimum,
        total_system_cost,
    )

    # -- 1. Build a random Erdos-Renyi directed graph --
    #    Fix 2: Smaller graph (n=10, p=0.3) so bandits have ~5-20 paths
    n = 10
    p = 0.3
    N_STEPS = 1000   # Fix 3: More steps for bandit learning
    N_AGENTS = 50
    print(f"Generating Erdos-Renyi graph  (n={n}, p={p}) ...")

    # Retry until we get a graph with a valid source->sink path
    G = None
    source, sink = 0, n - 1
    for attempt in range(50):
        seed = 42 + attempt
        G = nx.gnp_random_graph(n, p, seed=seed, directed=True)

        # Assign latency functions: l(x) = a*x + b
        # Store a, b explicitly so the Frank-Wolfe solver can use them
        rng = random.Random(seed)
        for u, v in G.edges():
            a = rng.uniform(0.5, 2.0)
            b = rng.uniform(0.1, 1.0)
            G[u][v]['a'] = a
            G[u][v]['b'] = b
            G[u][v]['latency'] = lambda x, _a=a, _b=b: _a * x + _b

        # Verify path exists
        if nx.has_path(G, source, sink):
            break
    else:
        raise RuntimeError(
            f"Could not find a connected ER graph with path {source}->{sink} "
            f"after 50 attempts.  Increase p or n."
        )

    paths = get_all_paths(G, source, sink)
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Paths (source={source} -> sink={sink}): {len(paths)}")

    # -- 2. Compute TRUE social-optimum cost via Frank-Wolfe --
    #    Fix 1: Use Experiment1's Frank-Wolfe solver instead of heuristic
    import copy
    G_fw = copy.deepcopy(G)
    social_flow, social_iters, _ = compute_social_optimum(
        G_fw, source, sink, demand=N_AGENTS,
        latency_type="linear", max_iter=500, tol=1e-8,
    )
    edges_fw = sorted(G_fw.edges())
    opt_cost = total_system_cost(G_fw, social_flow, edges_fw, "linear")
    print(f"  Social optimum cost (Frank-Wolfe, {social_iters} iters): {opt_cost:.2f}")

    # -- 3. Run mixed simulation (50% bandit) --
    print("\n" + "=" * 60)
    print("Mixed Simulation  (50% Thompson Sampling, 50% selfish)")
    print("=" * 60)
    sim_result = run_mixed_simulation(
        G, source, sink,
        n_agents=N_AGENTS,
        pct_bandit=0.5,
        n_steps=N_STEPS,
        opt_cost=opt_cost,
    )
    valid_poas = [p for p in sim_result['poa_history'] if p is not None]
    if valid_poas:
        print(f"  Final PoA (last 50 steps): {np.mean(valid_poas[-50:]):.4f}")

    # -- 4. Run PoA-vs-fraction experiment --
    print("\n" + "=" * 60)
    print("PoA vs Fraction of Thompson Sampling Agents")
    print("=" * 60)
    fraction_results = run_poa_vs_fraction_experiment(
        G, source, sink,
        n_agents=N_AGENTS,
        n_steps=N_STEPS,
        n_repeats=10,
        opt_cost=opt_cost,
    )

    # -- 5. Print summary table --
    print("\n" + "-" * 50)
    print(f"{'Fraction':>10s}  {'Mean PoA':>10s}  {'Std PoA':>10s}")
    print("-" * 50)
    for frac in sorted(fraction_results.keys()):
        r = fraction_results[frac]
        print(f"  {frac:>7.0%}    {r['mean_poa']:>10.4f}  {r['std_poa']:>10.4f}")
    print("-" * 50)

    # -- 6. Generate plots --
    print("\nGenerating plots ...")

    # Plot 1: PoA vs fraction
    plot_poa_vs_fraction(
        fraction_results,
        save_path=os.path.join(RESULTS_DIR, 'poa_vs_rl_fraction.png'),
    )

    # Plot 2: Convergence curves for a few representative fractions
    convergence_dict = {}
    for frac_label, frac_val in tqdm([('0% bandit', 0.0),
                                      ('25% bandit', 0.25),
                                      ('50% bandit', 0.5),
                                      ('100% bandit', 1.0)],
                                     desc="Convergence runs"):
        conv_sim = run_mixed_simulation(
            G, source, sink,
            n_agents=N_AGENTS,
            pct_bandit=frac_val,
            n_steps=N_STEPS,
            opt_cost=opt_cost,
        )
        convergence_dict[frac_label] = conv_sim['poa_history']

    plot_poa_convergence(
        convergence_dict,
        save_path=os.path.join(RESULTS_DIR, 'poa_convergence.png'),
    )

    print("\nDone.  Results saved to:", RESULTS_DIR)
