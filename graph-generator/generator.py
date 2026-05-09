from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx


@dataclass(frozen=True)
class GraphConfig:
    kind: str
    rows: int = 3
    cols: int = 3
    n: int = 10
    p: float = 0.2
    m: int = 2
    k: int = 4
    seed: Optional[int] = None
    directed: bool = True


def _maybe_directed(graph: nx.Graph, directed: bool) -> nx.Graph:
    if directed and not graph.is_directed():
        return graph.to_directed()
    return graph


def generate_grid(rows: int, cols: int, directed: bool = True) -> nx.Graph:
    graph = nx.grid_2d_graph(rows, cols)
    return _maybe_directed(graph, directed)


def generate_random(
    n: int, p: float, seed: Optional[int] = None, directed: bool = True
) -> nx.Graph:
    return nx.erdos_renyi_graph(n, p, seed=seed, directed=directed)


def generate_scale_free(
    n: int, m: int, seed: Optional[int] = None, directed: bool = True
) -> nx.Graph:
    graph = nx.barabasi_albert_graph(n, m, seed=seed)
    return _maybe_directed(graph, directed)


def generate_small_world(
    n: int, k: int, p: float, seed: Optional[int] = None, directed: bool = True
) -> nx.Graph:
    graph = nx.watts_strogatz_graph(n, k, p, seed=seed)
    return _maybe_directed(graph, directed)


def generate_complete(n: int, directed: bool = True) -> nx.Graph:
    graph = nx.complete_graph(n)
    return _maybe_directed(graph, directed)


def generate_graph(kind: str, **kwargs: object) -> nx.Graph:
    normalized = kind.strip().lower()

    if normalized in {"grid", "grid2d", "grid_2d"}:
        rows = int(kwargs.get("rows", 3))
        cols = int(kwargs.get("cols", 3))
        directed = bool(kwargs.get("directed", True))
        return generate_grid(rows, cols, directed=directed)

    if normalized in {"random", "erdos_renyi", "er"}:
        n = int(kwargs.get("n", 10))
        p = float(kwargs.get("p", 0.2))
        seed = kwargs.get("seed")
        directed = bool(kwargs.get("directed", True))
        return generate_random(n, p, seed=seed, directed=directed)

    if normalized in {"scale_free", "barabasi_albert", "ba"}:
        n = int(kwargs.get("n", 10))
        m = int(kwargs.get("m", 2))
        seed = kwargs.get("seed")
        directed = bool(kwargs.get("directed", True))
        return generate_scale_free(n, m, seed=seed, directed=directed)

    if normalized in {"small_world", "watts_strogatz", "ws"}:
        n = int(kwargs.get("n", 10))
        k = int(kwargs.get("k", 4))
        p = float(kwargs.get("p", 0.2))
        seed = kwargs.get("seed")
        directed = bool(kwargs.get("directed", True))
        return generate_small_world(n, k, p, seed=seed, directed=directed)

    if normalized in {"complete", "clique"}:
        n = int(kwargs.get("n", 10))
        directed = bool(kwargs.get("directed", True))
        return generate_complete(n, directed=directed)

    raise ValueError(f"Unsupported graph kind: {kind}")


def generate_from_config(config: GraphConfig) -> nx.Graph:
    return generate_graph(
        config.kind,
        rows=config.rows,
        cols=config.cols,
        n=config.n,
        p=config.p,
        m=config.m,
        k=config.k,
        seed=config.seed,
        directed=config.directed,
    )
