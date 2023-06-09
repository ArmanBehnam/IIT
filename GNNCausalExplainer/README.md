# Causal Graph Explainer

  Explaining neural networks helps to interpret the information within the network. Graph Neural Networks (GNNs) are a great architecture to study the neural networks explanation. In this work, we proposed the GNNCausalExplainer, a novel approach for GNN graph-level explanation by causality. Until now, all the neural networks' explanation have been based on association, but in this paper we get the information out of the dataset by calculation of expressivity of Neural Causal models (NCMs). We used intervention for making a model on NCM data structure instead of labels. The input of our framework is the dataset and the output is the explanatory subgraph $\Gamma$-GNN, it's causal model, and the information provided by this subgraph. Our approach contains three main steps: 1) Identifying neural causal identities, 2) the neural model estimation, and 3) explain the subgraph based on the model and evaluate that by it's expressivity. According to the experiments on synthetic and real-world graphs, our approach works well for neural identification of expressiveness and explaining graphs based on causal inference of that graph.

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

# NCM File

`ncm.py` provides the implementation of `SCM` (Structural Causal Model) and `NCM` (Neural Causal Model) classes.

## SCM Class

The `SCM` class represents a structural causal model, where each variable is a function of its parents in a causal graph and an independent noise term.

### Attributes

- `target_node`: The target node in the causal graph.
- `fn`: First hop neighborhood.
- `sn`: Second hop neighborhood.
- `on`: Out of hop neighborhood.
- `f`: Function.
- `u`: Exogenous variables.
- `v`: Endogenous variables.

### Methods

- `__init__(self, target_node, fn, sn, on)`: Initializes an `SCM` object.
- `size(self)`: Returns the size of each neighborhood and the total size.
- `space(self, i_size, select=None, tensor=True)`: Generates all possible states in the space.
- `intervention_fn(self)`: A random intervention function that adds a random value to `v` and `u`.
- `sample(self, n=None, u=None, do={}, select=None)`: Generates a sample from the SCM.
- `convert_evaluation(self, samples)`: Converts the samples.
- `forward(self, n=None, u=None, do={}, select=None, evaluating=False)`: Performs a forward pass through the SCM.
- `query_loss(self, input, val)`: Calculates the loss for a query.

## NCM Class

The `NCM` class is an extension of the `SCM` class and represents a Neural Causal Model.

### Attributes

- `cg`: A CausalGraph object.
- `u_size`: Size of the exogenous variables.
- `v_size`: Size of the endogenous variables.
- `f`: Function.
- `pu`: UniformDistribution object.

### Methods

- `__init__(self, cg, v_size={}, default_v_size=1, u_size={}, default_u_size=1, f={}, default_module=MLP)`: Initializes an `NCM` object.
- `biased_nll(self, v, n=1, do={})`: Computes the biased negative log likelihood.
- `nll(self, v, n=1, do={}, m=100000, alpha=80, return_biased=False)`: Computes the negative log likelihood.
- `nll_marg(self, v, n=1, m=10000, do={}, return_biased=False)`: Computes the marginal negative log likelihood.



