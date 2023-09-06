import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

def create_pos(G, layer_map):
    pos = {}
    layer_counts = {layer: 0 for layer in layer_map.values()}
    nodes_per_layer = {layer: 0 for layer in layer_map.values()}

    # Count the nodes in each layer
    for _, data in G.nodes(data=True):
        layer = layer_map[data['node_type']]
        nodes_per_layer[layer] += 1

    # Calculate positions
    for node, data in G.nodes(data=True):
        layer = layer_map[data['node_type']]
        layer_counts[layer] += 1
        if data["node_type"] == "arc":
            pos[node] = (-2 + 4 * layer_counts[layer] / (nodes_per_layer[layer] + 1), 0.5 + layer * 0.2)
        else:
            pos[node] = (-2 + 4 * layer_counts[layer] / (nodes_per_layer[layer] + 1), 0 + layer * 0.2)
    return pos

def plot_graph(hetero_data):
    
    color_map = {
        "city": "red",
        "arc": "yellow"
    }

    layer_map = {
        "city": 0,
        "arc": 1
    }
    G = nx.Graph()

    # Adding nodes to the graph
    node_counter = 0
    node_to_id_map = {}
    for node_type in hetero_data.node_types:
        type_counter = 0
        for node in hetero_data[node_type]["x"]:
            node_to_id_map[(node_type, type_counter)] = node_counter
            G.add_node(node_counter, node_type=node_type)
            node_counter += 1
            type_counter += 1
    
    arc_features = hetero_data["arc"]["x"]
    for i, feature in enumerate(arc_features):
        node_id = node_to_id_map[("arc", i)]
        raw_feature = feature.tolist()
        truncated_feature = math.ceil(raw_feature[0] * 10000) / 10000
        G.nodes[node_id]["features"] = truncated_feature

    # Adding edges to the graph
    for edge_type in hetero_data.edge_types:
        src_type, connection_type, dst_type = edge_type
        if "rev" in connection_type:
            continue
        edge_indices = hetero_data[edge_type].edge_index.t().tolist()  # get the edge indices
        for i, j in edge_indices:
            src_node = node_to_id_map[(src_type, i)]
            dst_node = node_to_id_map[(dst_type, j)]
            G.add_edge(src_node, dst_node)
    
    # Adding node labels (vectors)
    node_labels = {}
    for node_id, data in G.nodes(data=True):
        node_labels[node_id] = str(node_id)
        if data["node_type"] == "arc" and "features" in data:
            
            node_labels[node_id] +=  "\n " + str(data["features"])

    # Node colors mapping
    pos = create_pos(G, layer_map)
    node_colors = [color_map[data["node_type"]] for _, data in G.nodes(data=True)]
    legend_handles = [mpatches.Patch(color=color, label=node_type) for node_type, color in color_map.items()]
    plt.legend(handles=legend_handles)
    # Now use networkx's drawing function to visualize the graph
    nx.draw(G, pos=pos, node_color=node_colors, labels=node_labels, with_labels=True)
    plt.show()