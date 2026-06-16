"""
Cross-Agent Summary Table Generator
=====================================

Computes and saves a CSV summary table comparing all three bandit
algorithms across all three network topologies.

Output:
    results/comparison/cross_agent_summary.csv

Usage:
    python generate_summary_table.py
"""

import csv
import os

os.makedirs("results/comparison", exist_ok=True)

results = [
    # (algorithm, network, selfish_poa, final_poa, nash_type)
    ("ε-Greedy",          "Pigou",    1.3333, 1.2711, "Tight-bound"),
    ("ε-Greedy",          "Braess",   1.3333, 1.2913, "Tight-bound"),
    ("ε-Greedy",          "ER Graph", 1.0006, 1.0267, "Near-optimal"),
    ("UCB",               "Pigou",    1.3333, 1.1824, "Tight-bound"),
    ("UCB",               "Braess",   1.3333, 1.1291, "Tight-bound"),
    ("UCB",               "ER Graph", 1.0006, 2.0971, "Near-optimal"),
    ("Thompson Sampling", "Pigou",    1.3333, 1.2880, "Tight-bound"),
    ("Thompson Sampling", "Braess",   1.3333, 1.2614, "Tight-bound"),
    ("Thompson Sampling", "ER Graph", 1.0006, 1.0420, "Near-optimal"),
]

fieldnames = ["algorithm", "network", "selfish_poa", "final_poa",
              "poa_change", "pct_change", "direction", "nash_type"]

with open("results/comparison/cross_agent_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for alg, net, s_poa, f_poa, nash in results:
        change = round(f_poa - s_poa, 4)
        pct    = round((change / s_poa) * 100, 2)
        direction = "Improved" if change < 0 else "Degraded"
        writer.writerow({
            "algorithm":   alg,
            "network":     net,
            "selfish_poa": s_poa,
            "final_poa":   f_poa,
            "poa_change":  change,
            "pct_change":  pct,
            "direction":   direction,
            "nash_type":   nash,
        })

print("Saved: results/comparison/cross_agent_summary.csv")
