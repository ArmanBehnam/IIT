import itertools
import matplotlib.pyplot as plt
import causal
import torch.nn.functional as F
import example_scm
import torch as T
import torch.nn as nn
import numpy as np

cg = causal.CausalGraph(['A', 'B', 'C', 'D'], [('A', 'B'), ('A', 'C'), ('B', 'D')])
target_node, fn, sn, on = cg.categorize_neighbors()


class MLP(nn.Module):
    def __init__(self, fn_size, o_size, h_size=128, h_layers=2, use_sigmoid=True, use_layer_norm=True):
        self.fn_size = fn_size
        super().__init__()
        self.target_node = target_node
        self.fn = fn    # first hop neighborhood
        self.sn = sn    # second hop neighborhood
        self.on = on    # out of hop neighborhood
        self.set_fn = set(self.fn)
        self.u = self.sn.union(self.on) # exogenous variables
        self.fn_size = {v: 1 for v in self.fn}
        self.u_size = {v: 1 for v in self.u}
        self.o_size = o_size
        self.h_size = h_size
        self.i_size = sum(self.fn_size[k] for k in self.fn_size) + sum(self.u_size[k] for k in self.u_size)

        layers = [nn.Linear(self.i_size, self.h_size)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(self.h_size))
        layers.append(nn.ReLU())
        for l in range(h_layers - 1):
            layers.append(nn.Linear(self.h_size, self.h_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(self.h_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.h_size, self.o_size))
        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.nn = nn.Sequential(*layers)

        self.device_param = nn.Parameter(T.empty(0))

        self.nn.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            T.nn.init.xavier_normal_(m.weight, gain=T.nn.init.calculate_gain('relu'))

    def forward(self, fn, u, include_inp=False):
        assert isinstance(fn, dict), "fn should be a dictionary"
        assert isinstance(u, dict), "u should be a dictionary"
        assert all(isinstance(v, T.Tensor) for v in fn.values()), "All values in fn should be Tensors"
        assert all(isinstance(v, T.Tensor) for v in u.values()), "All values in u should be Tensors"

        if len(u.keys()) == 0:
            print({k: type(fn[k]) for k in self.fn if k in fn})

            inp = T.cat([fn[k] for k in self.fn if k in fn], dim=1) if len(self.fn) > 0 else T.zeros((1, 0))
        elif len(fn.keys()) == 0 or len(set(fn.keys()).intersection(self.set_fn)) == 0:
            inp = T.cat([u[k] for k in self.u], dim=1) if len(self.u) > 0 else T.zeros((1, 0))
        else:
            inp_fn = T.cat([fn[k] for k in self.fn if k in fn and fn[k].nelement() != 0], dim=1) if len(
                self.fn) > 0 else T.zeros((1, 0))
            inp_u = T.cat([u[k] for k in self.u if k in u and u[k].nelement() != 0], dim=1) if len(
                self.u) > 0 else T.zeros((1, 0))
            inp = T.cat((inp_fn, inp_u), dim=1)

        if include_inp:
            return self.nn(inp), inp

        return self.nn(inp)

    def binary_cross_entropy(self, output, target):
        return F.binary_cross_entropy(output, target)

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
        self.us = list(causal.CausalGraph(us, directed_edges=list(set((u2us[p], c)
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

if __name__ == "__main__":
    bernoulli_distribution = BernoulliDistribution(['U_A1', 'U_A2', 'U_AB', 'U_AC', 'U_B', 'U_C', 'U_D'],
                                                   {'U_A1': 1, 'U_A2': 1, 'U_AB': 1, 'U_AC': 1, 'U_B': 1, 'U_C': 1,
                                                    'U_D': 1},
                                                   p=0.5,
                                                   seed=0)
    scm = example_scm.SCM(bernoulli_distribution)
    # Training parameters
    epochs = 5
    batch_size = 32
    learning_rate = 0.01
    num_samples = 1000
    m1 = MLP(cg, o_size=1)
    # print("Results of simple MLP:", fn)
    # print(m1.fn_size, m1.u_size, m1.o_size, m1.h_size, m1.i_size)
    train_data = []
    optimizer = T.optim.Adam(m1.parameters(), lr=learning_rate)
    for _ in range(num_samples):
        sample = scm.generate_sample(intervention=np.random.randint(2))  # intervene on C by setting it to 1
        sample_tensors = {k: T.tensor([v], dtype=T.float32).view(1, -1) for k, v in sample.items() if k != 'A'}

        # Add the sample to the training data
        train_data.append((sample_tensors, T.tensor([sample['A']], dtype=T.float32)))
    estimated_probs =[]
    losses = []
    # Training loop
    for epoch in range(epochs):  # This is the number of epochs
        print(f"Epoch ", epoch)
        prob = []
        epoch_losses = []
        for fn, y in train_data:
            u = {key: value for key, value in fn.items() if key != 'A'}  # Exclude 'A' from u
            output = m1(fn, u)
            prob.append(float(output))
            loss = F.binary_cross_entropy(output, T.tensor([y], dtype=T.float32).view(-1, 1))
            loss.backward()
            epoch_losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
        print(prob)
        print(f"Mean P(A = 1 | C = 1): after epoch , {epoch} , {np.mean(prob)}")
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))
        # After training, you can estimate P(A=1|C=1) like this:
        test_fn = {
        'A': T.tensor([0.]).view(1, -1),
        'B': T.tensor([0.]).view(1, -1),
        'C': T.tensor([1.]).view(1, -1),
        'D': T.tensor([0.]).view(1, -1)
        }  # A test case where C=1
        test_u = {key: value for key, value in test_fn.items() if key != 'A'}  # Exclude 'A' from u
        prob_A_given_C = m1(test_fn, test_u).item()  # This is the estimated P(A=1|C=1)
        estimated_probs.append(prob_A_given_C)    # Plot the loss over epochs
        losses.append(np.mean(epoch_losses))
    plt.figure()
    plt.plot(range(epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')

    # Plot the estimated probability over epochs
    plt.figure()
    plt.plot(range(epochs), estimated_probs)
    plt.xlabel('Epoch')
    plt.ylabel('Estimated P(A=1|C=1)')
    plt.title('Estimated P(A=1|C=1) over epochs')

    plt.show()