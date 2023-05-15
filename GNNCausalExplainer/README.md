README.md
Overview
This repository contains Python scripts related to the analysis of causal graphs, represented by the CausalGraph class. This class is a tool for representing and manipulating causal graphs, which are data structures used to model causal relationships between different variables.

Main functionalities include:

Building a graph from a set of nodes and edges
Analyzing the graph structure
Searching for paths within the graph
Saving and loading a graph from a file
Visualizing the graph using the networkx library
Files
causal.py
This script contains the definition of the CausalGraph class, and a set of related utility functions.

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
