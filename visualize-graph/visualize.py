from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx


def _layout_for_graph(graph: nx.Graph, layout: str, seed: Optional[int]) -> dict:
	normalized = layout.strip().lower()
	if normalized in {"spring", "fr"}:
		return nx.spring_layout(graph, seed=seed)
	if normalized in {"kamada_kawai", "kk"}:
		return nx.kamada_kawai_layout(graph)
	if normalized in {"circular", "circle"}:
		return nx.circular_layout(graph)
	if normalized in {"shell"}:
		return nx.shell_layout(graph)
	if normalized in {"spectral"}:
		return nx.spectral_layout(graph)
	if normalized in {"grid", "grid2d"}:
		if all(isinstance(node, tuple) and len(node) == 2 for node in graph.nodes()):
			return {node: node for node in graph.nodes()}
		return nx.spring_layout(graph, seed=seed)
	raise ValueError(f"Unsupported layout: {layout}")


def visualize_graph(
	graph: nx.Graph,
	*,
	title: Optional[str] = None,
	layout: str = "spring",
	seed: Optional[int] = None,
	node_size: int = 500,
	node_color: str = "lightsteelblue",
	edge_color: str = "gray",
	font_size: int = 9,
	with_labels: bool = True,
	arrows: Optional[bool] = None,
	ax: Optional[plt.Axes] = None,
	show: bool = True,
) -> plt.Axes:
	if ax is None:
		_, ax = plt.subplots(figsize=(6, 4))

	positions = _layout_for_graph(graph, layout, seed)
	draw_arrows = graph.is_directed() if arrows is None else arrows

	nx.draw_networkx(
		graph,
		pos=positions,
		ax=ax,
		node_size=node_size,
		node_color=node_color,
		edge_color=edge_color,
		with_labels=with_labels,
		arrows=draw_arrows,
		font_size=font_size,
	)

	if title:
		ax.set_title(title)
	ax.axis("off")

	if show:
		plt.tight_layout()
		plt.show()

	return ax
