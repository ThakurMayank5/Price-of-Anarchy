# Empirical Findings — Cross-Algorithm Bandit Study
## NTCC Summer 2026 | Price of Anarchy in Congestion Games

---

## Finding 1 — Theoretical Bound Holds Across All Experiments

All empirical PoA values stayed within the range [1.0, ~2.10] across all three bandit algorithms and all three network topologies. On the tight-bound instances (Pigou and Braess networks), the 4/3 ≈ 1.333 theoretical bound was never violated — the selfish Nash equilibrium matched the bound exactly at 1.3333, and the introduction of bandit agents only moved the PoA downward. The only case where PoA exceeded 4/3 was UCB on the Erdős–Rényi random graph at high bandit fractions (reaching 2.0971 at 100%). This does not violate the Roughgarden–Tardos bound, because the ER graph's selfish Nash equilibrium is already near-optimal (PoA ≈ 1.0006); the 4/3 bound applies to the *worst-case* Nash equilibrium, not to a perturbed state where exploration noise pushes agents away from an efficient equilibrium.

---

## Finding 2 — PoA Impact Depends on Nash Equilibrium Quality (Central Finding)

The most important finding of this study is that the effect of bandit agents on system efficiency depends critically on the quality of the baseline Nash equilibrium. On tight-bound networks (Pigou and Braess), where the selfish Nash is maximally inefficient (PoA = 4/3), introducing bandit agents *improves* the system — the exploration component of bandit learning breaks agents out of the inefficient equilibrium and discovers lower-cost routing. Conversely, on the near-optimal ER random graph (selfish PoA ≈ 1.0006), bandit agents *degrade* the system because their exploration noise perturbs an already-efficient flow distribution. The mechanism is clear: exploration on an inefficient Nash equilibrium acts as a beneficial perturbation, while exploration on a near-optimal Nash equilibrium introduces unnecessary disruption.

---

## Finding 3 — Algorithm Ranking Differs by Network Type

The ranking of algorithms is not a single global order — it depends on both the network type and topology. On tight-bound networks, UCB is consistently the most effective algorithm. However, the ranking of ε-Greedy and Thompson Sampling differs by network: on Pigou, the order is **UCB > ε-Greedy > Thompson Sampling** (final PoA: 1.182 < 1.271 < 1.288); on Braess, it is **UCB > Thompson Sampling > ε-Greedy** (final PoA: 1.129 < 1.261 < 1.291). This reversal is itself a finding — ε-Greedy's fixed exploration rate is more effective on the simpler two-path Pigou topology, while Thompson Sampling's adaptive Bayesian exploration better navigates the three-path Braess network with its zero-cost shortcut. On the near-optimal ER graph, where minimal disruption is the goal, the ranking inverts entirely: **ε-Greedy ≈ Thompson Sampling ≪ UCB** (ε-Greedy causes only 2.6% degradation and Thompson Sampling 4.1%, while UCB causes a catastrophic 109.6% degradation). UCB is thus the highest-variance algorithm — most effective when improvement is needed, most destructive when the system is already efficient.

---

## Finding 4 — Thompson Sampling Offers the Best Trade-Off

Thompson Sampling emerges as the most balanced algorithm when considering performance across all network types. It achieves meaningful PoA reduction on tight-bound networks (Pigou: 1.333 → 1.288, a 3.4% improvement; Braess: 1.333 → 1.261, a 5.4% improvement) while causing only minimal disruption on the near-optimal ER graph (1.001 → 1.042, a 4.1% degradation). By comparison, UCB achieves stronger reduction on structured networks (up to 15.3% on Braess) but at the cost of catastrophic degradation on complex topologies (ER: 1.001 → 2.097, a 109.6% degradation). Thompson Sampling's Bayesian exploration mechanism naturally moderates the exploration-exploitation trade-off: as the posterior concentrates, exploration diminishes — preventing the persistent over-exploration that plagues UCB in near-optimal regimes.

---

## Finding 5 — Braess Paradox Resists Correction More Than Pigou

Comparing the two tight-bound networks reveals that the Braess network is harder to improve for ε-Greedy and Thompson Sampling, but *easier* for UCB. For ε-Greedy, Pigou shows a 4.7% PoA reduction at 100% while Braess shows only 3.2%. For Thompson Sampling, Pigou shows 3.4% reduction vs. Braess's 5.4%. The asymmetry arises from the zero-cost shortcut edge in the Braess network: this edge acts as a "trap" that continues to attract agents even during exploration, since its zero base cost makes it appear attractive under any exploration policy. UCB's aggressive exploration is forceful enough to overcome this trap, which explains why it achieves a stronger reduction on Braess (15.3%) than the other algorithms. The Braess network thus serves as a stress test for exploration strategies — only sufficiently aggressive exploration can counteract the paradoxical shortcut.

---

## Summary Table

| Algorithm | Network | Selfish PoA | Final PoA | PoA Change | % Change | Direction |  Nash Type |
|---|---|---|---|---|---|---|---|
| ε-Greedy | Pigou | 1.3333 | 1.2711 | -0.0622 | -4.67% | Improved | Tight-bound |
| ε-Greedy | Braess | 1.3333 | 1.2913 | -0.0420 | -3.15% | Improved | Tight-bound |
| ε-Greedy | ER Graph | 1.0006 | 1.0267 | +0.0261 | +2.61% | Degraded | Near-optimal |
| UCB | Pigou | 1.3333 | 1.1824 | -0.1509 | -11.32% | Improved | Tight-bound |
| UCB | Braess | 1.3333 | 1.1291 | -0.2042 | -15.31% | Improved | Tight-bound |
| UCB | ER Graph | 1.0006 | 2.0971 | +1.0965 | +109.59% | Degraded | Near-optimal |
| Thompson Sampling | Pigou | 1.3333 | 1.2880 | -0.0453 | -3.40% | Improved | Tight-bound |
| Thompson Sampling | Braess | 1.3333 | 1.2614 | -0.0719 | -5.39% | Improved | Tight-bound |
| Thompson Sampling | ER Graph | 1.0006 | 1.0420 | +0.0414 | +4.14% | Degraded | Near-optimal |

---

*Generated: Week 6 | 08/06/2026 – 14/06/2026*
*All values taken from simulation terminal output — Weeks 3, 4, 5*
