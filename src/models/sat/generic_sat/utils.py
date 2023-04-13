import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def hetero_data_to_networkx(data):
    G = nx.DiGraph()

    # Add nodes
    for node_type in data.node_types:
        for i in range(data[node_type].num_nodes):
            G.add_node(f"{node_type}_{i}", node_type=node_type)

    # Add edges
    for edge_key in data.edge_types:
        src, dst = data[edge_key].edge_index
        edge_type = edge_key[1]
        for s, d in zip(src, dst):
            G.add_edge(f"{edge_key[0]}_{s.item()}", f"{edge_key[2]}_{d.item()}", edge_type=edge_type)

    return G


def get_color_map(num_colors):
    color_map = plt.get_cmap('tab10')
    return [color_map(i) for i in np.linspace(0, 1, num_colors)]

def get_shape_list():
    return ['o', 's', '^', 'D', 'h', 'p', '*', '8', 'v']

def plot_heterogeneous_graph(data, align_nodes=True, spacing=1.5):
    G = hetero_data_to_networkx(data)
    if align_nodes:
        pos = {}
        type_counter = {}
        for node in G.nodes:
            node_type = G.nodes[node]['node_type']
            if node_type not in type_counter:
                type_counter[node_type] = 0
            pos[node] = (type_counter[node_type] * spacing, list(data.node_types).index(node_type) * spacing)
            type_counter[node_type] += 1
    else:
        pos = nx.spring_layout(G, seed=42)

    node_colors_map = dict(zip(data.node_types, get_color_map(len(data.node_types))))
    node_shapes_map = dict(zip(data.node_types, get_shape_list()))
    edge_colors_map = dict(zip([edge_key[1] for edge_key in data.edge_types], get_color_map(len(data.edge_types))))

    for node_type in data.node_types:
        node_list = [n for n in G.nodes if G.nodes[n]['node_type'] == node_type]
        nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=node_colors_map[node_type], node_shape=node_shapes_map[node_type])

    for edge_type, edge_color in edge_colors_map.items():
        edge_list = [(src, dst) for src, dst, data in G.edges(data=True) if data['edge_type'] == edge_type]
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=edge_color, arrows=True, connectionstyle='arc3,rad=0.2')

    nx.draw_networkx_labels(G, pos)
    plt.axis('off')
    plt.show()

def custom_layout(G, data, spacing=1.5):
    pos = {}
    type_counter = {}
    radius = spacing / 2
    for node in G.nodes:
        node_type = G.nodes[node]['node_type']
        if node_type not in type_counter:
            type_counter[node_type] = 0
        angle = 2 * np.pi * type_counter[node_type] / data[node_type].num_nodes
        x_offset = list(data.node_types).index(node_type) * spacing
        pos[node] = (radius * np.cos(angle) + x_offset, radius * np.sin(angle))
        type_counter[node_type] += 1
    return pos