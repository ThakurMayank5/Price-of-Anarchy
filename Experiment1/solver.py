"""
Frank-Wolfe Algorithm for Traffic Assignment
=============================================

Solves for:
  - Wardrop (User) Equilibrium: each user minimizes own travel time
  - System (Social) Optimum: central planner minimizes total system cost

Latency functions supported:
  - Linear:  l(f) = a * f + b
  - BPR:     l(f) = free_flow * (1 + 0.15 * (f / capacity)^4)

The algorithm iteratively:
  1. Computes edge weights from current flows
  2. Finds shortest path (all-or-nothing direction)
  3. Updates flows with diminishing step size 2/(k+2)
  4. Records convergence metrics each iteration
"""

import networkx as nx


# ─────────────────────────────────────────────────────────────────────
# Latency Functions
# ─────────────────────────────────────────────────────────────────────

def edge_latency(data, flow, latency_type):
    """
    Compute travel time on an edge given current flow.

    Linear:  l(f) = a * f + b
    BPR:     l(f) = free_flow * (1 + 0.15 * (f / capacity)^4)
    """
    if latency_type == "linear":
        return data["a"] * flow + data["b"]
    elif latency_type == "bpr":
        t0 = data["free_flow"]
        cap = data["capacity"]
        return t0 * (1.0 + 0.15 * (flow / cap) ** 4)
    raise ValueError(f"Unknown latency type: {latency_type}")


def edge_marginal_cost(data, flow, latency_type):
    """
    Marginal cost = l(f) + f * l'(f).
    Used as edge weights when computing the social optimum.

    Linear:  mc(f) = 2*a*f + b       (since l' = a)
    BPR:     mc(f) = t0 * (1 + 0.75 * (f/cap)^4)
    """
    if latency_type == "linear":
        return 2.0 * data["a"] * flow + data["b"]
    elif latency_type == "bpr":
        t0 = data["free_flow"]
        cap = data["capacity"]
        return t0 * (1.0 + 0.75 * (flow / cap) ** 4)
    raise ValueError(f"Unknown latency type: {latency_type}")


# ─────────────────────────────────────────────────────────────────────
# Objective Functions
# ─────────────────────────────────────────────────────────────────────

def beckmann_objective(graph, flow, edges, latency_type):
    """
    Beckmann potential: sum_e integral_0^{f_e} l_e(x) dx.
    This is the objective minimized by Wardrop Frank-Wolfe.

    Linear integral:  a*f^2/2 + b*f
    BPR integral:     t0*f + t0*0.03*f^5/c^4
    """
    total = 0.0
    for u, v in edges:
        data = graph[u][v]
        f = flow[(u, v)]
        if latency_type == "linear":
            total += data["a"] * f ** 2 / 2.0 + data["b"] * f
        elif latency_type == "bpr":
            t0 = data["free_flow"]
            cap = data["capacity"]
            total += t0 * f + t0 * 0.03 * f ** 5 / cap ** 4
    return total


def total_system_cost(graph, flow, edges, latency_type):
    """
    Total system cost: sum_e f_e * l_e(f_e).
    This is what the social optimum minimizes.
    """
    total = 0.0
    for u, v in edges:
        data = graph[u][v]
        f = flow[(u, v)]
        total += f * edge_latency(data, f, latency_type)
    return total


# ─────────────────────────────────────────────────────────────────────
# Frank-Wolfe Solver
# ─────────────────────────────────────────────────────────────────────

def frank_wolfe(graph, source, target, demand, latency_type,
                use_marginal=False, max_iter=300, tol=1e-6):
    """
    Frank-Wolfe algorithm for traffic assignment.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed graph with latency parameters on edges.
    source, target : node
        Origin-destination pair.
    demand : float
        Total traffic demand (number of agents).
    latency_type : str
        "linear" or "bpr".
    use_marginal : bool
        False -> Wardrop/Nash equilibrium (minimize Beckmann).
        True  -> Social optimum (minimize total system cost).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on max flow change.

    Returns
    -------
    flow : dict  {(u,v): flow_value}
    iterations : int
    convergence : list of dict  (per-iteration metrics)
    """
    edges = sorted(graph.edges())
    flow = {e: 0.0 for e in edges}
    convergence = []

    # Choose weight function: latency for Nash, marginal cost for Social
    cost_fn = edge_marginal_cost if use_marginal else edge_latency

    # ── Step 0: Initial all-or-nothing assignment on free-flow costs ──
    for u, v in edges:
        graph[u][v]["weight"] = cost_fn(graph[u][v], 0.0, latency_type)

    path = nx.shortest_path(graph, source, target, weight="weight")
    for u, v in zip(path[:-1], path[1:]):
        flow[(u, v)] = demand

    # ── Main Frank-Wolfe loop ──
    final_iter = 0
    for k in range(max_iter):
        # 1. Set edge weights from current flows
        for u, v in edges:
            d = graph[u][v]
            d["weight"] = cost_fn(d, flow[(u, v)], latency_type)

        # 2. Shortest path -> auxiliary (all-or-nothing) direction
        path = nx.shortest_path(graph, source, target, weight="weight")
        aux = {e: 0.0 for e in edges}
        for u, v in zip(path[:-1], path[1:]):
            aux[(u, v)] = demand

        # 3. Diminishing step size
        step = 2.0 / (k + 2.0)

        # 4. Update flows: f = f + step * (aux - f)
        max_change = 0.0
        for e in edges:
            new_f = flow[e] + step * (aux[e] - flow[e])
            max_change = max(max_change, abs(new_f - flow[e]))
            flow[e] = new_f

        # 5. Compute objective for convergence tracking
        if use_marginal:
            obj = total_system_cost(graph, flow, edges, latency_type)
        else:
            obj = beckmann_objective(graph, flow, edges, latency_type)

        rel_gap = max_change / max(demand, 1e-10)

        convergence.append({
            "iteration": k + 1,
            "objective_value": obj,
            "relative_gap": rel_gap,
            "step_size": step,
            "max_flow_change": max_change,
        })

        final_iter = k + 1

        # 6. Check convergence
        if max_change < tol:
            break

    return flow, final_iter, convergence


# ─────────────────────────────────────────────────────────────────────
# Convenience Wrappers
# ─────────────────────────────────────────────────────────────────────

def compute_wardrop_equilibrium(graph, source, target, demand, latency_type,
                                max_iter=300, tol=1e-6):
    """Compute Wardrop (Nash / User) equilibrium via Frank-Wolfe."""
    return frank_wolfe(graph, source, target, demand, latency_type,
                       use_marginal=False, max_iter=max_iter, tol=tol)


def compute_social_optimum(graph, source, target, demand, latency_type,
                           max_iter=300, tol=1e-6):
    """Compute social (system) optimum via Frank-Wolfe with marginal costs."""
    return frank_wolfe(graph, source, target, demand, latency_type,
                       use_marginal=True, max_iter=max_iter, tol=tol)


def compute_poa(graph, nash_flow, social_flow, edges, latency_type):
    """
    Price of Anarchy = Nash total cost / Social optimum total cost.
    Always >= 1.0.  Returns inf if social cost is zero.
    """
    nash_cost = total_system_cost(graph, nash_flow, edges, latency_type)
    social_cost = total_system_cost(graph, social_flow, edges, latency_type)
    if social_cost < 1e-15:
        return float("inf"), nash_cost, social_cost
    return nash_cost / social_cost, nash_cost, social_cost
