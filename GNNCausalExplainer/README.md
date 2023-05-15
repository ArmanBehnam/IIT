# Causal Graph Analysis

This repository provides Python scripts for the creation, analysis, and manipulation of causal graphs.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
  - [causal.py](#causalpy)
  - [mlp.py](#mlppy)
  - [ncm.py](#ncmpy)

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

# mlp.py

`mlp.py` implements the `MLP` class, a Multi-Layer Perceptron model in PyTorch, along with various distribution classes.

## MLP Class

The `MLP` class is a Multi-Layer Perceptron model that uses linear layers, optional layer normalization, and ReLU activation.

### Attributes

- `cg`: A `CausalGraph` object.
- `i_size`: Size of input.
- `u_size`: Size of exogenous variables.
- `o_size`: Size of output (default is 1).
- `h_size`: Size of hidden layers.
- `h_layers`: Number of hidden layers.
- `use_layer_norm`: A boolean indicating whether to use layer normalization.
- `nn`: The neural network defined as a `nn.Sequential` object.

### Methods

- `__init__(self, cg, i_size, u_size, o_size=1, h_size=128, h_layers=2, use_layer_norm=False)`: Initializes an `MLP` object.
- `init_weights(self, m)`: Initializes the weights of the model.
- `forward(self, fn, u,include_inp=False)`: Performs forward pass of the model.

## Distribution Class

The `Distribution` class is a base class for different types of distributions.

### Attributes

- `u`: A list of variables in the distribution.

### Methods

- `__init__(self, u)`: Initializes a `Distribution` object.
- `__iter__(self)`: Returns an iterator over the variables.
- `sample(self, n=1, device='cpu')`: Samples from the distribution (not implemented in base class).

