"""
Week 7 — Combined Learning Convergence Plots
=============================================

Generates cross-algorithm convergence plots showing how PoA evolves
over simulation steps for ε-Greedy, UCB, and Thompson Sampling — all
on the same axes, for both the Braess and Pigou networks.

Each algorithm is shown as a smoothed rolling mean (window=50 steps)
with a ±1 std shaded band across 5 independent seeds.

This is the "combined" version of the individual per-experiment
convergence plots that already exist in Experiment2-4/. Those plots
are one algorithm at a time, on a dark background, with raw step-level
noise. This script puts all three algorithms on one set of axes with
smoothing so the actual convergence trend is readable.

Outputs (in results/week7/):
    convergence_combined_braess.png
    convergence_combined_pigou.png

Usage:
    python generate_convergence_plots.py
    (must be run from repo root)
"""

import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Path setup ──
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
N_STEPS = 1000
N_SEEDS = 5        # independent runs per algorithm for error bands
PCT_BANDIT = 1.0   # 100% learners — clearest convergence signal
SMOOTH_WINDOW = 50  # rolling mean window size

ALGO_STYLES = {
    "ε-Greedy":          {"color": "#E07A5F", "linestyle": "-"},
    "UCB":               {"color": "#2E86AB", "linestyle": "-"},
    "Thompson Sampling": {"color": "#81B29A", "linestyle": "-"},
}
THEORETICAL_BOUND = 4 / 3


# ─────────────────────────────────────────────────────────────────────
# Simulation runner
# ─────────────────────────────────────────────────────────────────────

def collect_poa_histories(run_fn, build_fn, cost_fn, n_seeds=N_SEEDS):
    """
    Run `n_seeds` independent simulations, each with a fresh random seed,
    and return a (n_seeds × N_STEPS) array of per-step PoA values.
    """
    _, opt_cost, _ = cost_fn(N_AGENTS)
    histories = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        G, source, sink = build_fn(n_agents=N_AGENTS)
        result = run_fn(
            G, source, sink,
            n_agents=N_AGENTS,
            pct_bandit=PCT_BANDIT,
            n_steps=N_STEPS,
            opt_cost=opt_cost,
        )
        # replace None entries (warm-up steps before PoA is defined)
        # with the theoretical bound as a conservative placeholder
        history = [
            v if v is not None else THEORETICAL_BOUND
            for v in result["poa_history"]
        ]
        histories.append(history[:N_STEPS])

    return np.array(histories)  # shape: (n_seeds, N_STEPS)


def rolling_mean(arr, window):
    """Apply a causal rolling mean along axis=1 (steps axis)."""
    out = np.zeros_like(arr)
    for i in range(arr.shape[1]):
        start = max(0, i - window + 1)
        out[:, i] = arr[:, start:i + 1].mean(axis=1)
    return out


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_combined_convergence(ax, algo_histories, title):
    """
    Plot smoothed PoA convergence for all three algorithms on one axes.

    algo_histories : {algo_name: np.array shape (n_seeds, N_STEPS)}
    """
    steps = np.arange(1, N_STEPS + 1)

    for algo_name, histories in algo_histories.items():
        smoothed = rolling_mean(histories, SMOOTH_WINDOW)
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)

        style = ALGO_STYLES[algo_name]
        ax.plot(steps, mean,
                label=algo_name,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0)
        ax.fill_between(steps,
                        mean - std,
                        mean + std,
                        color=style["color"],
                        alpha=0.15)

    ax.axhline(THEORETICAL_BOUND, color="black", linestyle=":",
               linewidth=1.5, label=f"Theoretical bound (4/3 ≈ {THEORETICAL_BOUND:.3f})")
    ax.axhline(1.0, color="gray", linestyle="-",
               linewidth=0.8, alpha=0.5, label="Social optimum (PoA = 1)")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(f"Simulation step (smoothed over {SMOOTH_WINDOW}-step window)")
    ax.set_ylabel("Price of Anarchy")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)
    ax.set_xlim(1, N_STEPS)


def generate_for_network(build_fn, cost_fn, network_name, out_filename):
    print(f"\nRunning {N_SEEDS} seeds × 3 algorithms on {network_name} network ...")

    algo_configs = {
        "ε-Greedy":          (run_eg,  build_fn, cost_fn),
        "UCB":               (run_ucb, build_fn, cost_fn),
        "Thompson Sampling": (run_ts,  build_fn, cost_fn),
    }

    algo_histories = {}
    for algo_name, (run_fn, bf, cf) in algo_configs.items():
        print(f"  {algo_name} ...", end=" ", flush=True)
        algo_histories[algo_name] = collect_poa_histories(run_fn, bf, cf)
        final_means = algo_histories[algo_name][:, -50:].mean()
        print(f"final PoA (last 50 steps, mean across seeds): {final_means:.4f}")

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_combined_convergence(
        ax, algo_histories,
        title=(
            f"Learning Convergence — {network_name} Network\n"
            f"PoA over {N_STEPS} steps, 100% bandit agents, "
            f"smoothed (window={SMOOTH_WINDOW}), mean ± 1 std across {N_SEEDS} seeds"
        ),
    )
    fig.tight_layout()
    out_path = OUT_DIR / out_filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_for_network(
        build_braess_network,
        analytical_braess_costs,
        "Braess",
        "convergence_combined_braess.png",
    )
    generate_for_network(
        build_pigou_network,
        analytical_pigou_costs,
        "Pigou",
        "convergence_combined_pigou.png",
    )
    print("\nAll done.")
