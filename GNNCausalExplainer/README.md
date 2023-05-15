# Causal Graph Analysis

This repository provides Python scripts for the creation, analysis, and manipulation of causal graphs.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
  - [causal.py](#causalpy)

# causal.py

`causal.py` provides the implementation of a `CausalGraph` class and related functionalities. 

## CausalGraph Class

The `CausalGraph` class represents a causal graph, which is a graph where edges represent causal relationships between nodes.

### Attributes

- `v`: List of vertices in the graph.
- `fn`: First neighborhood. A dictionary where each key is a node, and its value is a set of its direct neighbors.
- `sn`: Second neighborhood. A dictionary where each key is a node, and its value is a set of its second-degree neighbors.
- `on`: Out of neighborhood. A dictionary where each key is a node, and its value is a set of nodes that are not in its neighborhood.
- `p`: Set of paths in the graph.
- `ue`: Set of unobserved edges in the graph.

### Methods

- `__init__(self, V, path=[], unobserved_edges=[])`: Initializes a `CausalGraph` object with vertices `V`, paths `path`, and unobserved edges `unobserved_edges`.
- `subgraph(self, V_sub, V_cut_back=None, V_cut_front=None)`: Returns a subgraph of the original graph.
- `plot(self)`: Plots the graph using NetworkX and matplotlib.
- `categorize_neighbors(self)`: Categorizes the neighbors of nodes into one-hop neighbors, two-hop neighbors, and out of neighborhood.
- `read(cls, filename)`: Class method that reads a `CausalGraph` object from a file.
- `save(self, filename)`: Saves the `CausalGraph` object to a file.

## graph_search Function

The `graph_search` function uses Breadth-First Search (BFS) to find a path between two nodes in a given `CausalGraph`. It supports searching for both directed and undirected paths.

### Parameters

- `cg`: A `CausalGraph` object.
- `v1`: The start node.
- `v2`: The end node. If `v2` is None, the function returns all nodes reachable from `v1`.
- `edge_type`: The type of edge to consider in the search. If `edge_type` is "path", the function searches for a path between `v1` and `v2`. If `edge_type` is "unobserved", the function searches for an unobserved path between `v1` and `v2`.

### Returns

- If `v2` is not None, the function returns a boolean indicating whether a path (observed or unobserved) exists between `v1` and `v2`.
- If `v2` is None, the function returns a set of all nodes reachable from `v1`.
