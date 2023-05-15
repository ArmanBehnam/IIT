# Causal Graph Analysis

This repository provides Python scripts for the creation, analysis, and manipulation of causal graphs.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
  - [causal.py](#causalpy)
- [Dependencies](#dependencies)
- [License](#license)

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/<your-github-username>/causal-graph-analysis.git

Class: CausalGraph
This class represents a causal graph. It takes as input a set of nodes, a set of directed edges (representing causal relationships), and a set of undirected edges (representing unobserved relationships).

It provides the following functionalities:

Neighbor Categorization: Categorizes the neighbors of each node into first neighborhood (direct neighbors), second neighborhood (neighbors of neighbors), and out of neighborhood.
Path Search: Searches for a path between two nodes in the graph.
Subgraph Extraction: Extracts a subgraph from the current graph.
Graph Visualization: Plots the graph using the networkx library.
Graph Serialization: Converts a graph into a serialized form that can be written to a file.
Graph Deserialization: Reads a graph from a file.
Function: graph_search
This function uses Breadth-First Search (BFS) to find a path between two nodes in a given CausalGraph. It supports searching for both directed and undirected paths.

How to use:
cg = CausalGraph(['A', 'B', 'C', 'D'], [('A', 'B'), ('A', 'C'), ('B', 'D')])

The CausalGraph can be created by providing a list of nodes and edges:
