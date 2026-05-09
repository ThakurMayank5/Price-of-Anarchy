import random

import matplotlib.pyplot as plt
import networkx as nx

DEMAND = 1.0


def generate_grid(rows, cols):
    return nx.grid_2d_graph(rows, cols).to_directed()


def generate_random(n, p, seed):
    return nx.erdos_renyi_graph(n, p, seed=seed, directed=True)


def generate_scale_free(n, m, seed):
    return nx.barabasi_albert_graph(n, m, seed=seed).to_directed()


def generate_small_world(n, k, p, seed):
    return nx.watts_strogatz_graph(n, k, p, seed=seed).to_directed()


def assign_edge_costs(graph, seed, a_range=(0.5, 2.0), b_range=(0.5, 2.0)):
    rng = random.Random(seed)
    for u, v in sorted(graph.edges()):
        graph[u][v]["a"] = rng.uniform(*a_range)
        graph[u][v]["b"] = rng.uniform(*b_range)


def pick_source_target(graph):
    for source in graph.nodes():
        reachable = nx.descendants(graph, source)
        if reachable:
            target = sorted(reachable)[0]
            return source, target
    return None, None


def generate_random_with_path(n, p, seed, max_tries=10):
    for attempt in range(max_tries):
        graph = generate_random(n, p, seed + attempt)
        source, target = pick_source_target(graph)
        if source is not None:
            return graph, source, target
    raise ValueError("No reachable source-target pair found; increase p or max_tries.")


def frank_wolfe(graph, source, target, demand=DEMAND, max_iter=120, tol=1e-6, use_marginal=False):
    edges = sorted(graph.edges())
    flow = {edge: 0.0 for edge in edges}
    iterations = 0

    for iteration in range(max_iter):
        # Update weights from current flows for shortest-path assignment.
        for u, v in edges:
            data = graph[u][v]
            edge_flow = flow[(u, v)]
            cost = data["a"] * edge_flow + data["b"]
            if use_marginal:
                cost += data["a"] * edge_flow
            data["weight"] = cost

        try:
            path = nx.shortest_path(graph, source=source, target=target, weight="weight")
        except nx.NetworkXNoPath as exc:
            raise ValueError("No path between source and target.") from exc

        aux = {edge: 0.0 for edge in edges}
        for u, v in zip(path[:-1], path[1:]):
            aux[(u, v)] = demand

        step = 2.0 / (iteration + 2.0)
        max_delta = 0.0
        for edge in edges:
            new_flow = flow[edge] + step * (aux[edge] - flow[edge])
            max_delta = max(max_delta, abs(new_flow - flow[edge]))
            flow[edge] = new_flow

        iterations = iteration + 1
        if max_delta < tol:
            break

    return flow, iterations


def total_cost(graph, flow):
    total = 0.0
    for (u, v), edge_flow in flow.items():
        data = graph[u][v]
        total += edge_flow * (data["a"] * edge_flow + data["b"])
    return total


def count_used_edges(flow, tol=1e-6):
    return sum(1 for edge_flow in flow.values() if edge_flow > tol)


def print_summary(name, graph, source, target, nash_flow, nash_iters, social_flow, social_iters):
    nash_cost = total_cost(graph, nash_flow)
    social_cost = total_cost(graph, social_flow)
    poa = float("inf") if social_cost == 0 else nash_cost / social_cost
    print(f"{name} graph:")
    print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    print(f"  Source: {source}, Target: {target}, Demand: {DEMAND}")
    print(
        f"  Nash cost: {nash_cost:.6f} (iters: {nash_iters}, used edges: {count_used_edges(nash_flow)})"
    )
    print(
        f"  Social cost: {social_cost:.6f} (iters: {social_iters}, used edges: {count_used_edges(social_flow)})"
    )
    print(f"  PoA: {poa:.6f}\n")
    return nash_cost, social_cost, poa


def layout_for_graph(name, graph, seed):
    if name == "Grid":
        return {node: node for node in graph.nodes()}
    return nx.spring_layout(graph, seed=seed)


def flow_widths(edges, flow, scale=6.0, min_width=0.6):
    max_flow = max(flow.values()) if flow else 0.0
    if max_flow <= 0:
        return [min_width] * len(edges)
    return [min_width + scale * (flow[edge] / max_flow) for edge in edges]


def draw_routing_subplot(ax, graph, pos, flow, title, source, target):
    edges = sorted(graph.edges())
    widths = flow_widths(edges, flow)
    show_labels = graph.number_of_nodes() <= 12
    node_colors = []
    for node in graph.nodes():
        if node == source:
            node_colors.append("gold")
        elif node == target:
            node_colors.append("tomato")
        else:
            node_colors.append("lightgray")

    nx.draw_networkx(
        graph,
        pos=pos,
        ax=ax,
        with_labels=show_labels,
        node_color=node_colors,
        edge_color="steelblue",
        width=widths,
        arrows=True,
        arrowsize=12,
        font_size=8,
    )
    ax.set_title(title)
    ax.axis("off")


def visualize_results(results, max_cols=3):
    for start in range(0, len(results), max_cols):
        chunk = results[start : start + max_cols]
        cols = len(chunk)
        fig, axes = plt.subplots(2, cols, figsize=(4.5 * cols, 8.0))
        if cols == 1:
            axes = [[axes[0]], [axes[1]]]

        for idx, result in enumerate(chunk):
            name = result["name"]
            graph = result["graph"]
            pos = result["pos"]
            source = result["source"]
            target = result["target"]

            draw_routing_subplot(
                axes[0][idx],
                graph,
                pos,
                result["nash_flow"],
                f"{name} Nash\nCost {result['nash_cost']:.3f}",
                source,
                target,
            )
            draw_routing_subplot(
                axes[1][idx],
                graph,
                pos,
                result["social_flow"],
                f"{name} Social\nCost {result['social_cost']:.3f}",
                source,
                target,
            )

        fig.suptitle("Routing Outcomes (edge width ~ flow)")
        plt.tight_layout()
        plt.show()


def run_demo():
    grid_small = generate_grid(3, 3)
    grid_large = generate_grid(5, 5)

    random_small, random_small_source, random_small_target = generate_random_with_path(
        n=10, p=0.2, seed=21
    )
    random_large, random_large_source, random_large_target = generate_random_with_path(
        n=20, p=0.15, seed=33
    )

    scale_small = generate_scale_free(n=10, m=2, seed=31)
    scale_large = generate_scale_free(n=20, m=3, seed=37)

    small_world = generate_small_world(n=20, k=4, p=0.3, seed=41)

    demos = [
        ("Grid 3x3", grid_small, (0, 0), (2, 2), 101),
        ("Grid 5x5", grid_large, (0, 0), (4, 4), 111),
        ("Random 10", random_small, random_small_source, random_small_target, 121),
        ("Random 20", random_large, random_large_source, random_large_target, 131),
        ("Scale-free 10", scale_small, 0, 9, 141),
        ("Scale-free 20", scale_large, 0, 19, 151),
        ("Small-world 20", small_world, 0, 19, 161),
    ]

    results = []
    for name, graph, source, target, cost_seed in demos:
        assign_edge_costs(graph, seed=cost_seed)
        nash_flow, nash_iters = frank_wolfe(
            graph, source, target, demand=DEMAND, max_iter=160, tol=1e-6, use_marginal=False
        )
        social_flow, social_iters = frank_wolfe(
            graph, source, target, demand=DEMAND, max_iter=160, tol=1e-6, use_marginal=True
        )
        nash_cost, social_cost, poa = print_summary(
            name, graph, source, target, nash_flow, nash_iters, social_flow, social_iters
        )
        results.append(
            {
                "name": name,
                "graph": graph,
                "source": source,
                "target": target,
                "pos": layout_for_graph(name, graph, seed=cost_seed + 7),
                "nash_flow": nash_flow,
                "social_flow": social_flow,
                "nash_cost": nash_cost,
                "social_cost": social_cost,
                "poa": poa,
            }
        )

    visualize_results(results)


if __name__ == "__main__":
    run_demo()
