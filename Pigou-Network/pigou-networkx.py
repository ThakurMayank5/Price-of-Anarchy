# Pigou Network using NetworkX

try:
    import networkx as nx
except ImportError as exc:
    raise SystemExit("Install networkx with: pip install networkx") from exc

DEMAND = 1.0


def build_pigou_network():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(["s", "t"])

    def cost_path1(flow):
        return flow

    def cost_path2(flow):
        return 1.0

    def cost_path1_derivative(flow):
        return 1.0

    def cost_path2_derivative(flow):
        return 0.0

    graph.add_edge(
        "s",
        "t",
        key="path1",
        name="path1",
        cost_func=cost_path1,
        cost_derivative=cost_path1_derivative,
    )
    graph.add_edge(
        "s",
        "t",
        key="path2",
        name="path2",
        cost_func=cost_path2,
        cost_derivative=cost_path2_derivative,
    )
    return graph


def edge_cost(graph, edge_key, flow):
    data = graph["s"]["t"][edge_key]
    return data["cost_func"](flow)


def edge_marginal_cost(graph, edge_key, flow):
    data = graph["s"]["t"][edge_key]
    return data["cost_func"](flow) + flow * data["cost_derivative"](flow)


def total_cost_from_path1(graph, flow_on_path1, demand=DEMAND):
    flow_on_path2 = demand - flow_on_path1
    return (
        flow_on_path1 * edge_cost(graph, "path1", flow_on_path1)
        + flow_on_path2 * edge_cost(graph, "path2", flow_on_path2)
    )


def wardrop_equilibrium(graph, demand=DEMAND, tol=1e-9, max_iter=100):
    def cost_gap(flow_on_path1):
        return edge_cost(graph, "path1", flow_on_path1) - edge_cost(
            graph, "path2", demand - flow_on_path1
        )

    if cost_gap(demand) <= 0:
        return demand
    if cost_gap(0.0) >= 0:
        return 0.0

    low, high = 0.0, demand
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        gap = cost_gap(mid)
        if abs(gap) <= tol:
            return mid
        if gap > 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


def social_optimum(graph, demand=DEMAND, tol=1e-9, max_iter=100):
    def marginal_gap(flow_on_path1):
        return edge_marginal_cost(graph, "path1", flow_on_path1) - edge_marginal_cost(
            graph, "path2", demand - flow_on_path1
        )

    if marginal_gap(demand) <= 0:
        return demand
    if marginal_gap(0.0) >= 0:
        return 0.0

    low, high = 0.0, demand
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        gap = marginal_gap(mid)
        if abs(gap) <= tol:
            return mid
        if gap > 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


def price_of_anarchy(graph, demand=DEMAND):
    flow_nash = wardrop_equilibrium(graph, demand)
    flow_opt = social_optimum(graph, demand)
    cost_nash = total_cost_from_path1(graph, flow_nash, demand)
    cost_opt = total_cost_from_path1(graph, flow_opt, demand)
    if cost_opt == 0:
        return float("inf")
    return cost_nash / cost_opt


if __name__ == "__main__":
    pigou_graph = build_pigou_network()
    flow_nash = wardrop_equilibrium(pigou_graph)
    flow_opt = social_optimum(pigou_graph)
    poa = price_of_anarchy(pigou_graph)

    print(f"Nash flow on path1: {flow_nash / DEMAND:.6f}")
    print(f"Optimal flow on path1: {flow_opt / DEMAND:.6f}")
    print(f"Price of Anarchy: {poa:.6f}")
