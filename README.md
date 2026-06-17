# Price of Anarchy in Congestion Games

An empirical study of the **Price of Anarchy (PoA)** in traffic routing games — combining classical Frank-Wolfe traffic assignment with multi-armed bandit reinforcement learning agents, across multiple network topologies (Pigou, Braess, grid, Erdős–Rényi, and Barabási-Albert scale-free graphs).


---

## What this project does

The Price of Anarchy measures how much worse a system performs when agents act selfishly (Nash/Wardrop equilibrium) compared to a centrally coordinated optimum (social optimum). 

This project:

1. **Computes Wardrop equilibria and social optima** on synthetic road networks using Frank-Wolfe traffic assignment, with both linear and BPR latency functions.
2. **Introduces bandit-learning agents** (ε-Greedy, UCB1, Thompson Sampling) into classic congestion-game test cases — Pigou and Braess networks — to study how individual learning dynamics shift the system away from or toward the selfish equilibrium.
3. **Measures how PoA responds** as the fraction of bandit-driven agents increases from 0% to 100%, and compares all three algorithms head-to-head.

## Key findings

The effect of introducing learning agents into a congestion game **depends entirely on how inefficient the starting equilibrium already is**:

- On **tight-bound networks** (Pigou, Braess), where the selfish equilibrium is already maximally inefficient (PoA = 4/3), bandit exploration nudges agents toward better routes — PoA **improves**. UCB is the strongest performer here, cutting PoA by up to **15.3%** on the Braess network.
- On a **near-optimal Erdős–Rényi random graph** (selfish PoA ≈ 1.0006), the same exploration becomes pure noise — PoA **degrades**. UCB's aggressive exploration is catastrophic here, causing a **109.6% degradation**, while ε-Greedy and Thompson Sampling stay nearly flat.
- **Thompson Sampling** offers the best overall trade-off: meaningful improvement on inefficient networks, minimal disruption on efficient ones — its Bayesian exploration naturally tapers off as its posterior concentrates.

Full writeup with all five findings and the complete results table: [`results/comparison/FINDINGS.md`](results/comparison/FINDINGS.md).

## Repository structure

```
.
├── Experiment1/              Frank-Wolfe traffic assignment across grid, ER, and
│                              scale-free topologies (10–100 nodes, sparse & dense)
│   ├── solver.py              Frank-Wolfe solver: Wardrop equilibrium + social optimum
│   ├── main.py                Experiment runner — generates graphs, demand, runs solver
│   ├── plots.py               Plotting utilities for convergence & PoA analysis
│   └── results/                Output CSV + generated plots
│
├── Pigou-Network/             Classic two-link Pigou network: brute-force,
│                              Nash equilibrium, social optimum, PoA computation
│
├── Experiment2/                ε-Greedy bandit agent on Pigou & Braess networks
├── Experiment3/                UCB1 bandit agent on Pigou & Braess networks
├── Experiment4/                Thompson Sampling (Normal-Inverse-Gamma prior) agent
│                                on Pigou & Braess networks
│
├── compare_all_agents.py      Generates cross-algorithm comparison plots
│                                (PoA vs. bandit-agent fraction) for all three topologies
├── generate_summary_table.py  Builds the summary results table
│
├── graph-generator/            Configurable graph generator (grid / random / scale-free)
├── visualize-graph/             NetworkX graph visualization helper
├── graphs/                       Graph generation scripts used across experiments
├── Routing-Demo/                 Standalone routing demo script
│
└── results/comparison/
    ├── FINDINGS.md              Full write-up of all empirical findings
    └── cross_agent_summary.csv  Final PoA values for every (algorithm × network) pair
```

## How it works

**Frank-Wolfe traffic assignment** (`Experiment1/solver.py`) iteratively computes both the Wardrop (user) equilibrium and the system (social) optimum:
1. Compute edge weights from current flow
2. Find the shortest path (all-or-nothing assignment)
3. Update flows with a diminishing step size of `2/(k+2)`
4. Track convergence each iteration

Supports linear (`l(f) = a·f + b`) and BPR (`l(f) = free_flow · (1 + 0.15·(f/capacity)^4)`) latency functions.

**Bandit agents** (`Experiment2`–`4`) are introduced as a fraction of the total agent population on the Pigou and Braess networks. Each agent learns which route to take through repeated interaction:
- **ε-Greedy** — explores a random route with fixed probability ε, otherwise exploits the best-known route
- **UCB1** — selects routes by an upper-confidence bound on estimated cost, balancing exploration and exploitation via uncertainty
- **Thompson Sampling** — maintains a Normal-Inverse-Gamma posterior over route costs and samples from it, naturally reducing exploration as confidence grows

## Setup

```bash
pip install networkx numpy pandas matplotlib tqdm
```

## Running the experiments

```bash
# Frank-Wolfe traffic assignment across all topologies
python Experiment1/main.py

# Pigou network: Nash equilibrium, social optimum, and PoA (brute-force baseline)
python Pigou-Network/price-of-anarchy.py

# Bandit agents on Pigou / Braess networks
python Experiment2/pigou_epsilon_greedy.py
python Experiment3/braess_ucb.py
python Experiment4/pigou_thompson.py

# Cross-algorithm comparison plots
python compare_all_agents.py

# Summary table
python generate_summary_table.py
```

## Tech stack

Python · NetworkX · NumPy · Pandas · Matplotlib · tqdm

## License

MIT — see [LICENSE](LICENSE). 
