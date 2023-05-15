import numpy as np
import networkx as nx
from torch.distributions import Bernoulli
import itertools

import torch as T
import torch.nn as nn

# create the graph
G = nx.DiGraph()
G.add_nodes_from(['A', 'B', 'C', 'D'])
G.add_edges_from([('A', 'B'), ('B', 'A'), ('A', 'C'), ('C', 'A'), ('B', 'D'), ('D', 'B')])

# define the functions for each node
f = {'A': lambda v, u: u['A'],
     'B': lambda v, u: T.stack([u['A'], u['B']]),
     'C': lambda v, u: u['C'],
     'D': lambda v, u: T.prod(T.stack([u['B'], u['D']])).item()}


# define the distribution over the exogenous variables
pu = {'A': Bernoulli(0.5),
      'B': Bernoulli(0.5),
      'C': Bernoulli(0.5),
      'D': Bernoulli(0.5)}

class SCM(nn.Module):
    def __init__(self, v, f, pu: dict):
        super().__init__()
        self.v = v
        self.u = list(pu)
        self.f = f
        self.pu = pu
        self.device_param = nn.Parameter(T.empty(0))

    def space(self, select=None, tensor=True):
        if select is None:
            select = self.v
        for pairs in itertools.product(*([
                (vi, T.LongTensor([[0]]).to(self.device_param.device) if tensor else 0),
                (vi, T.LongTensor([[1]]).to(self.device_param.device) if tensor else 1)]
                                         for vi in select)):
            yield dict(pairs)

    def forward(self, n=None, u=None, do={}, select=None):
        assert not set(do.keys()).difference(self.v)
        assert (n is None) != (u is None)
        if u is None:
            u = {k: self.pu[k].sample((n,)) for k in self.v}
        if select is None:
            select = self.v
        v = {}
        remaining = set(select)
        for k in self.v:
            v[k] = do[k] if k in do else self.f[k](v, u)
            remaining.discard(k)
            if not remaining:
                break
        return {k: v[k] for k in select}

if __name__ == "__main__":

    # create the SCM
    scm = SCM(list(G.nodes()), f, pu)

    # generate data from the SCM
    data = scm.forward(n=1000)

    # print the generated data
    print(data)