import pandas as pd
import numpy as np
from collections import defaultdict
from causal1 import CausalGraph
d = "D:/University/Spring2023/Research/Session22-230511/Codes/My/AIDS/"

# Load the data
# Load the data
edges_df = pd.read_csv(d + 'AIDS_A.txt', sep=',', header=None, names=['from', 'to'])
graph_indicator = pd.read_csv(d + 'AIDS_graph_indicator.txt', header=None, names=['graph_id'])
node_labels = pd.read_csv(d + 'AIDS_node_labels.txt', header=None, names=['node_label'])

# Add graph_id and node_label to edges_df
edges_df['graph_id'] = graph_indicator
edges_df['node_label'] = node_labels

# Create a dictionary where keys are graph_ids and values are lists of edges for that graph
edges_dict = defaultdict(list)

for idx, row in edges_df.iterrows():
    graph_id = row['graph_id']
    edge = (row['from'], row['to'])
    edges_dict[graph_id].append(edge)

# Now you can create a CausalGraph for each graph in the dataset
for graph_id, edges in edges_dict.items():
    vertices = np.unique(edges)
    cg = CausalGraph(vertices, edges)
    # Continue with your computations...

    # For instance, to print the neighborhoods for each graph
    target_node, one_hop_neighbors, two_hop_neighbors, out_of_neighborhood = cg.categorize_neighbors()
    print(f"Graph {graph_id}")
    print(f"Target node: {target_node}")
    print(f"1-hop neighbors of {target_node}: {one_hop_neighbors}")
    print(f"2-hop neighbors of {target_node}: {two_hop_neighbors}")
    print(f"Out of neighborhood of {target_node}: {out_of_neighborhood}")
    print("\n")