import torch as T
import torch.nn as nn
import numpy as np
import itertools
import causal1
class MLP(nn.Module):
    def __init__(self, cg, i_size, u_size, o_size=1, h_size=128, h_layers=2, use_layer_norm=False):
        super().__init__()
        self.cg = cg
        self.i_size = i_size    # size of input
        self.u_size = u_size    # size of exogenous variables
        self.o_size = o_size    # size of output
        self.h_size = h_size
        self.set_fn = set(self.fn)
        self.h_layers = h_layers
        self.use_layer_norm = use_layer_norm
        self.linear = nn.Linear(i_size, h_size)

        layers = [nn.Linear(self.u_size, self.h_size)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(self.h_size))
        layers.append(nn.ReLU())
        for l in range(h_layers - 1):
            layers.append(nn.Linear(self.h_size, self.h_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(self.h_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.h_size, self.o_size))

        self.nn = nn.Sequential(*layers)
        self.device_param = nn.Parameter(T.empty(0))
        # Apply init_weights only if i_size and u_size are not zero
        if self.i_size > 0 and self.u_size > 0:
            self.nn.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            T.nn.init.xavier_normal_(m.weight,gain=T.nn.init.calculate_gain('relu'))

    def forward(self, fn, u,include_inp=False):
        if len(u.keys()) == 0:
            inp = T.cat([fn[k] for k in self.fn], dim=1)
        elif len(fn.keys()) == 0 or len(set(fn.keys()).intersection(self.set_fn)) == 0:
            inp = T.cat([u[k] for k in u], dim=1)
        else:
            inp_u = T.cat([u[k] for k in u], dim=1)
            inp_fn = T.cat([fn[k] for k in self.fn], dim=1)
            inp = T.cat((inp_fn, inp_u), dim=1)

        if include_inp:
            return self.nn(inp), inp

        return self.nn(inp)

class Distribution(nn.Module):
    def __init__(self, u):
        super().__init__()
        self.u = u
        self.device_param = nn.Parameter(T.empty(0))

    def __iter__(self):
        return iter(self.u)

    def sample(self, n=1, device='cpu'):
        raise NotImplementedError()

    def forward(self, n=1):
        raise self.sample(n=n)


class DiscreteDistribution(Distribution):
    def __init__(self, u):
        super().__init__(u)

class FactorizedDistribution(DiscreteDistribution):
    def simplex_init(s):
        t = T.rand(s).flatten()
        t[-1] = 1
        t = t.sort()[0]
        t[1:] = t[1:] - t[:-1]
        return T.log(t.reshape(s))

    def __init__(self, us, cond={}, init='simplex'):
        us = list(map(lambda s: (s,) if type(
            s) is str else tuple(sorted(s)), us))
        super().__init__(list(ui for u in us for ui in u))
        if not all(k in self.u for vals in cond.values() for k in vals):
            raise ValueError('cond contains variables not in u')

        # sort us in topological order
        u2us = {u: next(usi for usi in us if u in usi) for u in self.u}
        self.us = list(causal1.CausalGraph(us, path=list(set((u2us[p], c)
                                                               for c in cond
                                                               for p in cond[c]))))
        self.cond = {u: sorted(cond.get(u, [])) for u in self.us}
        self.init = {
            'uniform': lambda s: T.zeros(s),  # wrong
            'simplex': FactorizedDistribution.simplex_init,
        }.get(init, init)
        self.q = nn.ParameterDict({
            str(us): nn.Parameter(self.init(tuple(2 for ui in itertools.chain(
                self.cond[us], us))))
            for us in self.us})

    def sample(self, n=1, device=None):
        if device is None:
            device = self.device_param.device
        qs = {u: (self.q[str(u)][None].to(device)
                  + -T.log(-T.log(T.rand((n,) + tuple(self.q[str(u)].shape),
                                         device=device))))
              for u in self.us}
        qs = {}
        for us in self.us:
            t = self.q[str(us)]

            # select conditional probability distribution
            if self.cond[us]:
                t = t[tuple(qs[u2].flatten() for u2 in self.cond[us])]
            else:
                t = t.expand((n,) + tuple(t.shape))

            # sample using Gumbel-max
            for i in range(10):  # in case there are two maximums in one row
                # Gumbel-max
                gm = t + -T.log(-T.log(T.rand(t.shape, device=device)))
                gm = ((gm == (gm.view(n, -1).max(dim=1).values
                              .reshape((n,) + (1,) * (len(gm.shape) - 1))))
                      .nonzero(as_tuple=False)[:, 1:])
                if len(gm) == n:
                    break
            else:
                raise ValueError(
                    f'something went wrong! gm has shape {gm.shape}')

            # split samples into variables
            qs.update({us[i]: gm[:, i:i+1] for i in range(len(us))})
        return qs
class BernoulliDistribution(DiscreteDistribution):
    def __init__(self, u_names, sizes, p, seed=None):
        assert set(sizes.keys()).issubset(set(u_names))

        super().__init__(list(u_names))
        self.sizes = {U: sizes[U] if U in sizes else 1 for U in u_names}
        self.p = p
        if seed is not None:
            self.rand_state = np.random.RandomState(seed=seed)
        else:
            self.rand_state = np.random.RandomState()

    def sample(self, n=1, device=None):
        if device is None:
            device = self.device_param.device

        u_vals = dict()
        for U in self.sizes:
            u_vals[U] = T.from_numpy(self.rand_state.binomial(1, self.p, size=(n, self.sizes[U]))).long().to(device)

        return u_vals


class SplitBernoulliDistribution(DiscreteDistribution):
    def __init__(self, u1_names, u2_names, sizes, p1, p2, seed=None):
        all_u_names = set(u1_names + u2_names)

        assert set(sizes.keys()).issubset(all_u_names)

        super().__init__(list(all_u_names))
        self.u1_names = u1_names
        self.u2_names = u2_names
        self.sizes = {U: sizes[U] if U in sizes else 1 for U in all_u_names}
        self.p1 = p1
        self.p2 = p2
        if seed is not None:
            self.rand_state = np.random.RandomState(seed=seed)
        else:
            self.rand_state = np.random.RandomState()

    def sample(self, n=1, device=None):
        if device is None:
            device = self.device_param.device

        u_vals = dict()
        for U in self.u1_names:
            u_vals[U] = T.from_numpy(self.rand_state.binomial(1, self.p1, size=(n, self.sizes[U]))).long().to(device)
        for U in self.u2_names:
            u_vals[U] = T.from_numpy(self.rand_state.binomial(1, self.p2, size=(n, self.sizes[U]))).long().to(device)

        return u_vals


class ContinuousDistribution(Distribution):
    def __init__(self, u):
        super().__init__(u)


class UniformDistribution(ContinuousDistribution):
    def __init__(self, u_names, sizes, seed=None):
        assert set(sizes.keys()).issubset(set(u_names))

        super().__init__(list(u_names))
        self.sizes = {U: sizes[U] if U in sizes else 1 for U in u_names}
        if seed is not None:
            self.rand_state = np.random.RandomState(seed=seed)
        else:
            self.rand_state = np.random.RandomState()

    def sample(self, n=1, device=None):
        if device is None:
            device = self.device_param.device

        u_vals = dict()
        for U in self.sizes:
            u_vals[U] = T.from_numpy(self.rand_state.rand(n, self.sizes[U])).float().to(device)

        return u_vals


class NeuralDistribution(ContinuousDistribution):
    def __init__(self, u_names, sizes, hyperparams, default_module=MLP, seed=None):
        assert set(sizes.keys()).issubset(set(u_names))

        super().__init__(list(u_names))
        self.sizes = {U: sizes[U] if U in sizes else 1 for U in u_names}
        if seed is not None:
            self.rand_state = np.random.RandomState(seed=seed)
        else:
            self.rand_state = np.random.RandomState()

        self.func = T.nn.ModuleDict({
            str(u): default_module(
                {},
                {u: self.sizes[u]},
                self.sizes[u],
                h_layers=hyperparams.get('h-layers', 2),
                h_size=hyperparams.get('h-size', 128),
                use_layer_norm=hyperparams.get('layer-norm', False),
                use_sigmoid=False
            )
            for u in self.sizes})

    def sample(self, n=1, device=None):
        if device is None:
            device = self.device_param.device

        u_vals = dict()
        for U in self.sizes:
            noise = T.randn((n, self.sizes[U])).float().to(device)
            u_vals[U] = self.func[str(U)]({}, {U: noise})

        return u_vals

