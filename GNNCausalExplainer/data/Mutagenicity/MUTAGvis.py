import os
import random
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
import networkx as nx
import matplotlib.pyplot as plt

# To visualize MUTAG dataset, we need to load our edges, nodes, and adjacency list first.
mut_edge = np.loadtxt('MUTAG_edge_labels.txt')
mut_node = np.loadtxt('MUTAG_node_labels.txt')
A = np.loadtxt('MUTAG_A.txt', dtype=str)

# Second, let’s define our nodes as a dictionary of node labels and a color that represent them.
# define colors for each label
color_dict = {0:"Red", 1:"Blue", 2:"Yellow",3:"Green", 4:"Gray", 5:"Purple", 6:"Pink"}
# create dict for nodes
node = {}
for i in range(200):
   node[str(i)] = dict(color=color_dict[mut_node[i]])

# For edges, we define a list of edges based on the adjacency list and their labels.
# dict for edge labels
bond_dict = {0:"aromatic", 1:"single", 2:"double", 3:"triple"}
# list to define edges
edges = []
for i in range(400):
   edges.append((A[i][0].split(',')[0], A[i][1],bond_dict[mut_edge[i]]))


# Last but not least, let’s create our network and start adding nodes, edges, and style.
# After all, we can just draw it out and see the result.
# init graph
original_graph = nx.Graph()
# add nodes and edges
original_graph.add_nodes_from(n for n in node.items())
original_graph.add_edges_from((u, v, {"type": label}) for u, v, label in edges)
# format plot
plt.figure(1, figsize=(25, 25), dpi=60)
# set baseline network plot options
base_options = dict(with_labels=False, edgecolors="black", node_size=200)
# define layout of position
pos = nx.spring_layout(original_graph, seed=7482934)
# set node's color as we define previously
node_colors = [d["color"] for _, d in original_graph.nodes(data=True)]
# add weights on edges to classify different bonds by widths
edge_type_visual_weight_lookup = {"aromatic":1,"single":2, "double":3, "triple":4}
edge_weights = [edge_type_visual_weight_lookup[d["type"]] for _, _, d in original_graph.edges(data=True)]
# draw network
nx.draw_networkx(original_graph, pos=pos, node_color=node_colors, width=edge_weights, **base_options)
plt.show()