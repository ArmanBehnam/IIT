import os
import random
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
import matplotlib.pyplot as plt

# set seed
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# set data path
DATA_PATH = "/home"

# load MUTAG dataset
dataset = TUDataset(root=DATA_PATH, name='MUTAG')


# calculate graph density
total_edges = 0
total_nodes = 0
for data in dataset:
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    total_edges += edge_index.shape[1]
    total_nodes += num_nodes

density = total_edges / (total_nodes * (total_nodes - 1))
# The number of edges divided by the number of possible edges

print("len:", len(dataset))
print("num_classes:", dataset.num_classes)
print("num_node_features:", dataset.num_node_features)
print("total_edges:", total_edges)
print("total_nodes:", total_nodes)
print("Graph density:", density)
print("How many random nodes should be added to the small subgraph with 5 nodes", density * 5*100)
print("How many random nodes should be added to the large subgraph with 500 nodes", density * 500*100)
# visualize the graph
data = dataset[0]
edge_index = data.edge_index
num_nodes = data.num_nodes
adj_matrix = torch.zeros((num_nodes, num_nodes))
adj_matrix[edge_index[0], edge_index[1]] = 1
plt.spy(adj_matrix)
plt.title("MUTAG Graph Density")
plt.show()
