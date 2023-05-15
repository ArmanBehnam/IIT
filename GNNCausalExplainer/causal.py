import re
import itertools
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

class CausalGraph:
    def __init__(self, V, path=[], unobserved_edges=[]):
        self.v = list(V)
        self.set_v = set(V)
        self.fn = {v: set() for v in V}  # First neighborhood
        self.sn = {v: set() for v in V}  # Second neighborhood
        self.on = {v: set() for v in V}  # Out of neighborhood

        self.p = set(map(tuple, map(sorted, path)))  # Path to First neighborhood
        self.ue = set(map(tuple, map(sorted, unobserved_edges)))  # Unobserved edges

        for v1, v2 in path:
            self.fn[v1].add(v2)
            self.fn[v2].add(v1)
            self.p.add(tuple(sorted((v1, v2))))

        self.categorize_neighbors()

        self._sort()
        self.v2i = {v: i for i, v in enumerate(self.v)}

        self.cc = self._c_components()
        self.v2cc = {v: next(c for c in self.cc if v in c) for v in self.v}
        self.pap = {
            v: sorted(set(itertools.chain.from_iterable(
                list(self.fn[v2]) + [v2]
                for v2 in self.v2cc[v]
                if self.v2i[v2] <= self.v2i[v])) - {v},
                      key=self.v2i.get)
            for v in self.v}
        self.c2 = self._maximal_cliques()
        self.v2c2 = {v: [c for c in self.c2 if v in c] for v in self.v}

    def __iter__(self):
        return iter(self.v)

    def subgraph(self, V_sub, V_cut_back=None, V_cut_front=None):
        assert V_sub.issubset(self.set_v)

        if V_cut_back is None:
            V_cut_back = set()
        if V_cut_front is None:
            V_cut_front = set()

        assert V_cut_back.issubset(self.set_v)
        assert V_cut_front.issubset(self.set_v)

        new_p = [(V1, V2) for V1, V2 in self.p
                  if V1 in V_sub and V2 in V_sub and V2 not in V_cut_back and V1 not in V_cut_front]
        new_ue = [(V1, V2) for V1, V2 in self.ue
                  if V1 in V_sub and V2 in V_sub and V1 not in V_cut_back and V2 not in V_cut_back]

        return CausalGraph(V_sub, new_p, new_ue)

    def _sort(self):
        # Topological sorting using DFS
        L = []
        marks = {v: 0 for v in self.v}  # 0: unmarked, 1: temporary, 2: permanent

    def _c_components(self):
        pool = set(self.v)
        cc = []
        while pool:
            cc.append({pool.pop()})
            while True:
                added = {k2 for k in cc[-1] for k2 in self.sn[k]}
                delta = added - cc[-1]
                cc[-1].update(delta)
                pool.difference_update(delta)
                if not delta:
                    break
        return [tuple(sorted(c, key=self.v2i.get)) for c in cc]

    def _maximal_cliques(self):
        # find degeneracy ordering
        o = []
        p = set(self.v)
        while len(o) < len(self.v):
            v = min((len(set(self.sn[v]).difference(o)), v) for v in p)[1]
            o.append(v)
            p.remove(v)

        # brute-force bron_kerbosch algorithm
        c2 = set()

        def bron_kerbosch(r, p, x):
            if not p and not x:
                c2.add(tuple(sorted(r)))
            p = set(p)
            x = set(x)
            for v in list(p):
                bron_kerbosch(r.union({v}),
                              p.intersection(self.sn[v]),
                              x.intersection(self.sn[v]))
                p.remove(v)
                x.add(v)

        # apply brute-force bron_kerbosch with degeneracy ordering
        p = set(self.v)
        x = set()
        for v in o:
            bron_kerbosch({v},
                          p.intersection(self.sn[v]),
                          x.intersection(self.sn[v]))
            p.remove(v)
            x.add(v)

        return c2

    def categorize_neighbors(self):
        centrality = {v: len(self.fn[v]) for v in self.v}
        target_node = max(centrality, key=centrality.get)

        if target_node not in self.set_v:
            return

        one_hop_neighbors = self.fn[target_node]
        two_hop_neighbors = set()

        for neighbor in one_hop_neighbors:
            two_hop_neighbors |= self.fn[neighbor]

        two_hop_neighbors -= one_hop_neighbors
        two_hop_neighbors.discard(target_node)
        out_of_neighborhood = self.set_v - (one_hop_neighbors | two_hop_neighbors | {target_node})

        self.sn[target_node] = two_hop_neighbors
        self.on[target_node] = out_of_neighborhood
        return target_node, one_hop_neighbors, two_hop_neighbors, out_of_neighborhood

    def plot(self):
        G = nx.Graph()
        G.add_nodes_from(self.v)
        G.add_edges_from(self.p)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=2000, font_size=20, font_weight='bold', node_color="lightblue", edge_color="grey")
        plt.savefig('d_causal.png')

    def convert_set_to_sorted(self, C):
        return [v for v in self.v if v in C]

    def serialize(self, C):
        return tuple(self.convert_set_to_sorted(C))

    @classmethod
    def read(cls, filename):
        with open(filename) as file:
            mode = None
            V = []
            directed_edges = []
            bidirected_edges = []
            try:
                for i, line in enumerate(map(str.strip, file), 1):
                    if line == '':
                        continue

                    m = re.match('<([A-Z]+)>', line)
                    if m:
                        mode = m.groups()[0]
                        continue

                    if mode == 'NODES':
                        if line.isidentifier():
                            V.append(line)
                        else:
                            raise ValueError('invalid identifier')
                    elif mode == 'EDGES':
                        if '<->' in line:
                            v1, v2 = map(str.strip, line.split('<->'))
                            bidirected_edges.append((v1, v2))
                        elif '->' in line:
                            v1, v2 = map(str.strip, line.split('->'))
                            directed_edges.append((v1, v2))
                        else:
                            raise ValueError('invalid edge type')
                    else:
                        raise ValueError('unknown mode')
            except Exception as e:
                raise ValueError(f'Error parsing line {i}: {e}: {line}')
            return cls(V, directed_edges, bidirected_edges)

    def save(self, filename):
        with open(filename, 'w') as file:
            lines = ["<NODES>\n"]
            for V in self.v:
                lines.append("{}\n".format(V))
            lines.append("\n")
            lines.append("<EDGES>\n")
            for V1, V2 in self.de:
                lines.append("{} -> {}\n".format(V1, V2))
            for V1, V2 in self.be:
                lines.append("{} <-> {}\n".format(V1, V2))
            file.writelines(lines)


def graph_search(cg, v1, v2=None, edge_type="path"):
    """
    Uses BFS to check for a path between v1 and v2 in cg. If v2 is None, returns all reachable nodes.
    """
    assert edge_type in ["path", "unobserved"]
    assert v1 in cg.set_v
    assert v2 in cg.set_v or v2 is None

    target_node, one_hop_neighbors, two_hop_neighbors, out_of_neighborhood = cg.categorize_neighbors()

    q = deque([v1])
    seen = {v1}
    while len(q) > 0:
        cur = q.popleft()
        cur_fn = cg.fn[cur]
        cur_sn = cg.sn[target_node]
        cur_on = cg.on[target_node]

        cur_neighbors = cur_fn if edge_type == "path" else (cur_sn | cur_on)

        for neighbor in cur_neighbors:
            if neighbor not in seen:
                if v2 is not None:
                    if (neighbor == v2 and edge_type == "path" and neighbor in one_hop_neighbors) or \
                            (neighbor == v2 and edge_type == "unobserved" and neighbor in (
                                    two_hop_neighbors | out_of_neighborhood)):
                        return True
                seen.add(neighbor)
                q.append(neighbor)

    if v2 is None:
        return seen

    return False

if __name__ == "__main__":
    cg = CausalGraph(['A', 'B', 'C', 'D'], [('A', 'B'), ('A', 'C'), ('B', 'D')])
    # cg = "My/cg/backdoor.cg"
    # Categorize neighbors for node A
    target_node, one_hop_neighbors, two_hop_neighbors, out_of_neighborhood = cg.categorize_neighbors()

    print(f"Target node: {target_node}")
    print(f"1-hop neighbors of A: {one_hop_neighbors}")
    print(f"2-hop neighbors of A: {two_hop_neighbors}")
    print(f"Out of neighborhood of A: {out_of_neighborhood}")
    # Example usage of graph_search
    result1 = graph_search(cg, 'A', 'D', edge_type="path")
    print(f"Is there a path from A to D? {result1}")

    result2 = graph_search(cg, 'A', 'D', edge_type="unobserved")
    print(f"Can there be an unobserved path from A to D? {result2}")

    result3 = graph_search(cg, 'A', edge_type="path")
    print(f"All nodes reachable from A via paths: {result3}")
    cg.plot()
