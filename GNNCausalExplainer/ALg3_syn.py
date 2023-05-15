import d_causal
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import math
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

def clique(start, nb_nodes, nb_to_remove=0, role_start=0):
    """ Defines a clique (complete graph on nb_nodes nodes,
    with nb_to_remove  edges that will have to be removed),
    index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    nb_nodes    :    int correspondingraph to the nb of nodes in the clique
    role_start  :    starting index for the roles
    nb_to_remove:    int-- numb of edges to remove (unif at RDM)
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    a = np.ones((nb_nodes, nb_nodes))
    np.fill_diagonal(a, 0)
    graph = nx.from_numpy_matrix(a)
    edge_list = graph.edges().keys()
    roles = [role_start] * nb_nodes
    if nb_to_remove > 0:
        lst = np.random.choice(len(edge_list), nb_to_remove, replace=False)
        print(edge_list, lst)
        to_delete = [edge_list[e] for e in lst]
        graph.remove_edges_from(to_delete)
        for e in lst:
            print(edge_list[e][0])
            print(len(roles))
            roles[edge_list[e][0]] += 1
            roles[edge_list[e][1]] += 1
    mapping_graph = {k: (k + start) for k in range(nb_nodes)}
    graph = nx.relabel_nodes(graph, mapping_graph)
    name = 'clique'
    return graph, roles, name

def cycle(start, len_cycle, role_start=0):
    """Builds a cycle graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + len_cycle))
    for i in range(len_cycle - 1):
        graph.add_edges_from([(start + i, start + i + 1)])
    graph.add_edges_from([(start + len_cycle - 1, start)])
    roles = [role_start] * len_cycle
    name = 'cycle'
    return graph, roles, name

def diamond(start, role_start=0):
    """Builds a diamond graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 12))
    graph.add_edges_from(
        [
            (start, start + 1),
            (start + 1, start + 2),
            (start + 2, start + 3),
            (start + 3, start),
        ]
    )
    graph.add_edges_from(
        [
            (start + 4, start),
            (start + 4, start + 1),
            (start + 4, start + 2),
            (start + 4, start + 3),
        ]
    )
    graph.add_edges_from(
        [
            (start + 5, start),
            (start + 5, start + 1),
            (start + 5, start + 2),
            (start + 5, start + 3),
            (start + 6, start + 4),
            (start + 6, start + 1),
            (start + 6, start + 2),
            (start + 7, start + 1),
            (start + 7, start + 2),
            (start + 8, start + 3),
            (start + 8, start + 3),
            (start + 9, start + 3),
            (start + 10, start + 3),
            (start + 10, start + 8),
            (start + 11, start + 3),
            (start + 11, start + 7),
            (start + 12, start + 11),
            (start + 12, start + 10),

        ]
    )
    roles = [role_start] * 12
    name = 'diamond'
    return graph, roles, name

def tree(start, height, r=2, role_start=0):
    """Builds a balanced r-tree of height h
    INPUT:
    -------------
    start       :    starting index for the shape
    height      :    int height of the tree
    r           :    int number of branches per node
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a tree shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at role_start)
    """
    graph = nx.balanced_tree(r, height)
    roles = [0] * graph.number_of_nodes()
    name = 'tree'
    return graph, roles, name


def fan(start, nb_branches, role_start=0):
    """Builds a fan-like graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    nb_branches :    int correspondingraph to the nb of fan branches
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph, roles, name = star(start, nb_branches, role_start=role_start)
    for k in range(1, nb_branches - 1):
        roles[k] += 1
        roles[k + 1] += 1
        graph.add_edges_from([(start + k, start + k + 1)])
    name = 'fan'
    return graph, roles, name

def ba(start, width, role_start=0, m=5):
    """Builds a BA preferential attachment graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    width       :    int size of the graph
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.barabasi_albert_graph(width, m)
    graph.add_nodes_from(range(start, start + width))
    nids = sorted(graph)
    mapping = {nid: start + i for i, nid in enumerate(nids)}
    graph = nx.relabel_nodes(graph, mapping)
    roles = [role_start for i in range(width)]
    name = 'ba'
    return graph, roles, name

def house(start, role_start=0):
    """Builds a house-like  graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 5))
    graph.add_edges_from(
        [
            (start, start + 1),
            (start + 1, start + 2),
            (start + 2, start + 3),
            (start + 3, start),
        ]
    )
    # graph.add_edges_from([(start, start + 2), (start + 1, start + 3)])
    graph.add_edges_from([(start + 4, start), (start + 4, start + 1)])
    roles = [role_start, role_start, role_start + 1, role_start + 1, role_start + 2]
    name = 'house'
    return graph, roles, name

def grid(start, dim=2, role_start=0):
    """ Builds a 2by2 grid
    """
    grid_G = nx.grid_graph([dim, dim])
    grid_G = nx.convert_node_labels_to_integers(grid_G, first_label=start)
    roles = [role_start for i in grid_G.nodes()]
    name = 'grid'
    return grid_G, roles, name


def star(start, nb_branches, role_start=0):
    """Builds a star graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    nb_branches :    int correspondingraph to the nb of star branches
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + nb_branches + 1))
    for k in range(1, nb_branches + 1):
        graph.add_edges_from([(start, start + k)])
    roles = [role_start + 1] * (nb_branches + 1)
    roles[0] = role_start
    name = 'star'
    return graph, roles, name


def path(start, width, role_start=0):
    """Builds a path graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    width       :    int length of the path
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + width))
    for i in range(width - 1):
        graph.add_edges_from([(start + i, start + i + 1)])
    roles = [role_start] * width
    roles[0] = role_start + 1
    roles[-1] = role_start + 1
    name = 'path'
    return graph, roles, name


def build_graph(
    width_basis,
    basis_type,
    list_shapes,
    start=0,
    rdm_basis_plugins=False,
    add_random_edges=0,
    m=5,
):
    """This function creates a basis (scale-free, path, or cycle)
    and attaches elements of the type in the list randomly along the basis.
    Possibility to add random edges afterwards.
    INPUT:
    --------------------------------------------------------------------------------------
    width_basis      :      width (in terms of number of nodes) of the basis
    basis_type       :      (torus, string, or cycle)
    shapes           :      list of shape list (1st arg: type of shape,
                            next args:args for building the shape,
                            except for the start)
    start            :      initial nb for the first node
    rdm_basis_plugins:      boolean. Should the shapes be randomly placed
                            along the basis (True) or regularly (False)?
    add_random_edges :      nb of edges to randomly add on the structure
    m                :      number of edges to attach to existing node (for BA graph)
    OUTPUT:
    --------------------------------------------------------------------------------------
    basis            :      a nx graph with the particular shape
    role_ids         :      labels for each role
    plugins          :      node ids with the attached shapes
    """
    if basis_type == "ba":
        basis, role_id = eval(basis_type)(start, width_basis, m=m)
    else:
        basis, role_id = eval(basis_type)(start, width_basis)

    n_basis, n_shapes = nx.number_of_nodes(basis), len(list_shapes)
    start += n_basis  # indicator of the id of the next node

    # Sample (with replacement) where to attach the new motifs
    if rdm_basis_plugins is True:
        plugins = np.random.choice(n_basis, n_shapes, replace=False)
    else:
        spacing = math.floor(n_basis / n_shapes)
        plugins = [int(k * spacing) for k in range(n_shapes)]
    seen_shapes = {"basis": [0, n_basis]}

    for shape_id, shape in enumerate(list_shapes):
        shape_type = shape[0]
        args = [start]
        if len(shape) > 1:
            args += shape[1:]
        args += [0]
        graph_s, roles_graph_s = eval(shape_type)(*args)
        n_s = nx.number_of_nodes(graph_s)
        try:
            col_start = seen_shapes[shape_type][0]
        except:
            col_start = np.max(role_id) + 1
            seen_shapes[shape_type] = [col_start, n_s]
        # Attach the shape to the basis
        basis.add_nodes_from(graph_s.nodes())
        basis.add_edges_from(graph_s.edges())
        basis.add_edges_from([(start, plugins[shape_id])])
        if shape_type == "cycle":
            if np.random.random() > 0.5:
                a = np.random.randint(1, 4)
                b = np.random.randint(1, 4)
                basis.add_edges_from([(a + start, b + plugins[shape_id])])
        temp_labels = [r + col_start for r in roles_graph_s]
        # temp_labels[0] += 100 * seen_shapes[shape_type][0]
        role_id += temp_labels
        start += n_s

    if add_random_edges > 0:
        # add random edges between nodes:
        for p in range(add_random_edges):
            src, dest = np.random.choice(nx.number_of_nodes(basis), 2, replace=False)
            print(src, dest)
            basis.add_edges_from([(src, dest)])

    return basis, role_id, plugins

def generate_and_plot_graphs():
    # Generate the clique graph
    graph1, roles1, name1 = clique(0, 10, nb_to_remove=0, role_start=0)
    # Generate the cycle graph
    graph2, roles2, name2 = cycle(0, 10)
    # Generate the diamond graph
    graph3, roles3, name3 = diamond(0, 0)
    # Generate the tree graph
    graph4, roles4, name4 = tree(0, 3, r=3, role_start=0)
    # Generate the fan graph
    graph5, roles5, name5 = fan(0, 20, role_start=0)
    # Generate the ba graph
    graph6, roles6, name6 = ba(0, 10, role_start=0, m=5)
    # Generate the grid graph
    graph7, roles7, name7 = grid(0, 5, role_start=0)


    graphs = [graph1, graph2, graph3, graph4, graph5, graph6, graph7]
    roles = [roles1, roles2, roles3, roles4, roles5, roles6, roles7]
    names = [name1, name2, name3, name4, name5, name6, name7]


    num_graphs = len(graphs)
    num_cols = 3
    num_rows = 3

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 15))

    for i, (graph, role) in enumerate(zip(graphs, roles)):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col] if num_rows > 1 else axs[col]

        pos = nx.spring_layout(graph)
        nx.draw_networkx_edges(graph, pos, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(graph, pos, ax=ax)
        nx.draw(graph, pos, with_labels=True, node_size=200, font_size=10, font_weight='bold', node_color="lightblue", edge_color="grey")
        ax.set_title(f"Graph {i+1}: {names[i]}")

    plt.tight_layout()
    plt.savefig('ALg3_syn.png')


def adjacency_matrices():
    graph_types = [
        ("clique", (0, 5)),
        ("cycle", (0, 5)),
        ("diamond", (0,)),
        ("tree", (0, 3)),
        ("fan", (0, 5)),
        ("ba", (0, 10)),
        ("house", (0,)),
        ("grid", (0, 2)),
        ("star", (0, 5)),
        ("path", (0, 5))
    ]
    adj = {}
    edges = {}
    for graph_type, args in graph_types:
        graph, roles, name = eval(graph_type)(*args)
        adjacency_matrix = nx.to_numpy_matrix(graph)
        print(f"{name.capitalize()} graph adjacency matrix:")
        print(adjacency_matrix)
        print("\n")
        adj[graph_type] = nx.adjacency_matrix(graph).todense().A.tolist()
        edges[graph_type] = set()
        for node, neighbors in enumerate(adj[graph_type]):
            for neighbor, edge in enumerate(neighbors):
                if edge:
                    edges[graph_type].add(tuple(sorted((node, neighbor))))
    return adj, edges


if __name__ == "__main__":
    generate_and_plot_graphs()
    adj, edges = adjacency_matrices()
    # Create CausalGraph objects for each graph
    for graph_type in adj:
        graph_cg = d_causal.CausalGraph(list(range(len(adj[graph_type]))), list(edges[graph_type]))
        target_node, one_hop_neighbors, two_hop_neighbors, out_of_neighborhood = graph_cg.categorize_neighbors()
        print(f"{graph_type.capitalize()} graph:")
        print(f"Target (Most important) node: {target_node}")
        print(f"1-hop neighbors of Target: {one_hop_neighbors}")
        print(f"2-hop neighbors of Target: {two_hop_neighbors}")
        print(f"Out of neighborhood of Target: {out_of_neighborhood}")
        result = d_causal.graph_search(graph_cg, target_node, edge_type="path")
        print(f"All nodes reachable from target node via paths: {result}\n")
