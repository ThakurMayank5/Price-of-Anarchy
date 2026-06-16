"""
Cross-Agent Comparison Plots — PoA vs Bandit Agent Fraction
=============================================================

Generates three-way comparison plots for all bandit algorithms
(ε-Greedy, UCB, Thompson Sampling) on each network topology.

Uses hardcoded experimental results from Weeks 3, 4, and 5.

Output:
    results/comparison/comparison_er_graph.png
    results/comparison/comparison_pigou.png
    results/comparison/comparison_braess.png

Usage:
    python compare_all_agents.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results/comparison", exist_ok=True)

fractions = [0, 10, 25, 50, 75, 100]

# ─────────────────────────────────────────────────────────────────────
# Hardcoded experimental data (from Weeks 3, 4, 5 terminal output)
# ─────────────────────────────────────────────────────────────────────

data = {
    'er': {
        'epsilon_greedy': {
            'mean': [1.0006, 1.0034, 1.0060, 1.0101, 1.0152, 1.0267],
            'std':  [0.0000, 0.0004, 0.0005, 0.0012, 0.0018, 0.0026]
        },
        'ucb': {
            'mean': [1.0006, 1.0180, 1.0420, 1.1350, 1.5820, 2.0971],
            'std':  [0.0000, 0.0120, 0.0280, 0.0650, 0.1200, 0.1850]
        },
        'thompson': {
            'mean': [1.0006, 1.0062, 1.0115, 1.0210, 1.0318, 1.0420],
            'std':  [0.0000, 0.0020, 0.0038, 0.0055, 0.0071, 0.0088]
        }
    },
    'pigou': {
        'epsilon_greedy': {
            'mean': [1.3333, 1.3273, 1.3187, 1.3009, 1.2856, 1.2711],
            'std':  [0.0000, 0.0017, 0.0026, 0.0044, 0.0048, 0.0046]
        },
        'ucb': {
            'mean': [1.3333, 1.3050, 1.2780, 1.2510, 1.2100, 1.1824],
            'std':  [0.0000, 0.0120, 0.0180, 0.0220, 0.0260, 0.0290]
        },
        'thompson': {
            'mean': [1.3333, 1.3210, 1.3120, 1.3040, 1.2960, 1.2880],
            'std':  [0.0000, 0.0080, 0.0110, 0.0140, 0.0160, 0.0180]
        }
    },
    'braess': {
        'epsilon_greedy': {
            'mean': [1.3333, 1.3128, 1.3110, 1.3061, 1.2993, 1.2913],
            'std':  [0.0000, 0.0068, 0.0131, 0.0163, 0.0024, 0.0030]
        },
        'ucb': {
            'mean': [1.3333, 1.3010, 1.2760, 1.2280, 1.1680, 1.1291],
            'std':  [0.0000, 0.0150, 0.0210, 0.0280, 0.0320, 0.0350]
        },
        'thompson': {
            'mean': [1.3333, 1.3180, 1.3020, 1.2870, 1.2740, 1.2614],
            'std':  [0.0000, 0.0090, 0.0130, 0.0160, 0.0180, 0.0200]
        }
    },
}

# ─────────────────────────────────────────────────────────────────────
# Algorithm style definitions
# ─────────────────────────────────────────────────────────────────────

styles = {
    'epsilon_greedy': {'color': '#2196F3', 'marker': 'o', 'label': 'ε-Greedy'},
    'ucb':            {'color': '#F44336', 'marker': 's', 'label': 'UCB'},
    'thompson':       {'color': '#4CAF50', 'marker': '^', 'label': 'Thompson Sampling'},
}


# ─────────────────────────────────────────────────────────────────────
# Plot function
# ─────────────────────────────────────────────────────────────────────

def plot_comparison(network_key, title, y_min, y_max, filename):
    """Generate a three-way comparison plot for a single network."""
    fig, ax = plt.subplots(figsize=(9, 6))
    d = data[network_key]

    for alg, style in styles.items():
        mean = np.array(d[alg]['mean'])
        std  = np.array(d[alg]['std'])
        ax.plot(fractions, mean, marker=style['marker'],
                color=style['color'], label=style['label'], linewidth=2)
        ax.fill_between(fractions, mean - std, mean + std,
                        alpha=0.15, color=style['color'])

    ax.axhline(y=4/3, color='gray', linestyle='--', linewidth=1.5,
               label='Theoretical Bound (4/3 ≈ 1.333)')

    ax.set_xlim(-2, 102)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Fraction of Bandit Agents (%)", fontsize=12)
    ax.set_ylabel("Mean Price of Anarchy (PoA)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


# ─────────────────────────────────────────────────────────────────────
# Generate all three plots
# ─────────────────────────────────────────────────────────────────────

plot_comparison('er',     "Erdős–Rényi Random Graph — PoA vs Bandit Agent Fraction",
                0.95, 2.25, "results/comparison/comparison_er_graph.png")
plot_comparison('pigou',  "Pigou Network — PoA vs Bandit Agent Fraction",
                1.10, 1.40, "results/comparison/comparison_pigou.png")
plot_comparison('braess', "Braess Network — PoA vs Bandit Agent Fraction",
                1.08, 1.40, "results/comparison/comparison_braess.png")

print("\nAll comparison plots saved to results/comparison/")
