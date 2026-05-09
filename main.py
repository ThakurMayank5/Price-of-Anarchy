from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence


ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "graph-generator"))
sys.path.insert(0, str(ROOT_DIR / "visualize-graph"))

from generator import GraphConfig, generate_from_config
from visualize import visualize_graph


def format_edges(edges: Sequence[tuple[object, object]], limit: Optional[int]) -> list[str]:
    if limit is None:
        limit = len(edges)
    return [str(edge) for edge in edges[:limit]]


def main() -> None:

    config = GraphConfig(
        kind="random",
        n=10,
        p=0.3,
        seed=42,
        directed=True,
    )

    graph = generate_from_config(config)
    print(f"Graph kind: {config.kind}")
    print(
        f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}, "
        f"Directed: {graph.is_directed()}"
    )

    visualize_graph(graph, title=f"{config.kind.capitalize()} Graph", layout="spring")



if __name__ == "__main__":
    main()
