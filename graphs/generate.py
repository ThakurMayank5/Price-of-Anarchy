import networkx as nx

import matplotlib.pyplot as plt

def generate_grid(rows, cols):
    return nx.grid_2d_graph(rows, cols).to_directed()

def generate_random(n, p):
    return nx.erdos_renyi_graph(n, p, directed=True)

def generate_scale_free(n, m):
    return nx.barabasi_albert_graph(n, m).to_directed()

if __name__ == "__main__":
    grid_graph = generate_grid(3, 3)
    random_graph = generate_random(10, 0.2)
    scale_free_graph = generate_scale_free(10, 2)

    print("Grid Graph:")
    print(grid_graph.edges())
    
    print("\nRandom Graph:")
    print(random_graph.edges())
    
    print("\nScale-Free Graph:")
    print(scale_free_graph.edges())

    # Visualize the graphs
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    nx.draw(grid_graph, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Grid Graph")

    plt.subplot(1, 3, 2)

    pos = nx.spring_layout(random_graph)

    nx.draw(random_graph, pos=pos, with_labels=True, node_color='lightgreen', edge_color='gray')
    plt.title("Random Graph")

    plt.subplot(1, 3, 3)
    nx.draw(scale_free_graph, with_labels=True, node_color='lightcoral', edge_color='gray')
    plt.title("Scale-Free Graph")
    plt.show()

    # G = nx.erdos_renyi_graph(15, 0.2, directed=True)

    # pos = nx.spring_layout(G, seed=42)

    # flows = []
    # for edge in G.edges():
    #     flows.append(G.edges[edge].get("flow", 1))

    # nx.draw(
    #     G,
    #     pos,
    #     with_labels=True,
    #     width=flows,
    #     arrows=True
    # )

    # plt.show()