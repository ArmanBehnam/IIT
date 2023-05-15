import itertools
import numpy as np
import torch as T
import torch.nn as nn
from mlp2 import *
from causal1 import CausalGraph
from utils import *
import random

class SCM(nn.Module):
    def __init__(self, target_node, fn, sn, on):
        super().__init__()
        self.target_node = target_node
        self.fn = fn    # first hop neighborhood
        self.sn = sn    # second hop neighborhood
        self.on = on    # out of hop neighborhood
        self.f = MLP   # function
        self.u = self.sn.union(self.on) # exogenous variables
        self.v = self.fn.union(self.sn) # exogenous variables


    def size(self):
        fn_size = len(self.fn)
        sn_size = len(self.sn)
        on_size = len(self.on)
        u_size = len(self.u) if self.u else 0
        i_size = fn_size + sn_size
        return fn_size, sn_size, on_size, u_size, i_size

    def space(self, i_size, select=None, tensor=True):
        if select is None:
            select = self.target_node
        for pairs in itertools.product(*([
            (vi, torch.LongTensor(value).to(self.device) if tensor else value)
            for value in itertools.product(*([0, 1] for _ in range(i_size[vi])))]
                for vi in select)):
            yield dict(pairs)

    def intervention_fn(self):
        """A random intervention function that adds a random value to `v` and `u`."""
        # For each key, value pair in `v` and `u`, add a random value from a normal distribution.
        print(self.u,self.v)
        for key in self.v:
            self.v[key] += random.gauss(0, 1)
        for key in self.u:
            self.u[key] += random.gauss(0, 1)
        return self.v,self.u

    def sample(self, n=None, u=None, do={}, select=None):
        assert not set(do.keys()).difference(self.target_node)
        assert (n is None) != (u is None)

        for k in do:
            do[k] = do[k].to(self.device)

        if select is None:
            select = self.target_node
        v = {}
        remaining = set(select)
        for k in self.fn:
            print(fn)
            v[k] = do[k] if k in do else self.intervention_fn()[k]
            remaining.discard(k)
            if not remaining:
                break
        return {k: v[k] for k in select}

    def convert_evaluation(self, samples):
        return samples

    def forward(self, n=None, u=None, do={}, select=None, evaluating=False):
        if evaluating:
            with torch.no_grad():
                result = self.sample(n, u, do, select)
                result = self.convert_evaluation(result)
                return {k: result[k].cpu() for k in result}

        return self.sample(n, u, do, select)

    def query_loss(self, input, val):
        if torch.is_tensor(val):
            raise NotImplementedError()
        else:
            if val == 1:
                return torch.sum(-torch.log(input))
            elif val == 0:
                return torch.sum(-torch.log(1 - input))
            else:
                raise ValueError(f"Comparison to {val} of type {type(val)} is not allowed.")

class NCM(SCM):
    def __init__(self, cg, v_size={}, default_v_size=1, u_size={},
                 default_u_size=1, f={}, default_module=MLP):
        target_node, fn, sn, on = cg.categorize_neighbors()
        super().__init__(target_node, fn, sn, on)
        self.cg = cg
        self.u_size = {k: u_size.get(k, default_u_size) for k in self.cg.c2}
        self.v_size = {k: v_size.get(k, default_v_size) for k in self.cg}
        self.f = nn.ModuleDict({
                k: f[k] if k in f else default_module(
                    {k: self.v_size[k] for k in self.cg.fn[k]},
                    {k: self.u_size[k] for k in self.cg.v2c2[k]},
                    self.v_size[k],
                )
                for k in cg})
        self.pu=UniformDistribution(self.cg.c2, {k: u_size.get(k, 1) for k in self.cg.c2})


    def biased_nll(self, v, n=1, do={}):
        assert not set(do.keys()).difference(self.v)
        mode = self.training
        try:
            self.train()
            batch_size = len(next(iter(v.values())))
            u = {k: t.expand((batch_size,) + tuple(t.shape)).transpose(0, 1)
                 for k, t in self.pu.sample(n=n).items()}  # (n, batch_size, var_size)
            v_new = {k: t.expand((n,) + t.shape).float()
                     for k, t in v.items()}
            logpv = 0
            for k in self.v:
                if k in do:
                    if do[k] != v[k]:
                        return float('-inf')
                else:
                    logpv += self.f[k](v_new, u, v_new[k])
            logpv = T.logsumexp(logpv, dim=0) - np.log(n)
            return -logpv
        finally:
            self.train(mode=mode)

    def nll(self, v, n=1, do={}, m=100000, alpha=80, return_biased=False):
        assert not set(do.keys()).difference(self.v)
        mode = self.training
        try:
            self.train()
            batch_size = len(next(iter(v.values())))

            # sample n Ks per batch (batch_size, n)
            uk = np.random.rand(batch_size, n)
            K = np.where(uk > 1 / alpha,
                         np.floor(1 / uk),
                         np.floor(np.log(alpha * uk) / np.log(0.9) + alpha))

            # compute log probabilities (batch_size, max_sum_n_samples)
            n_samples = K + (m - 1)
            max_sum_n_samples = int(n_samples.sum(axis=1).max())
            u = {k: t.reshape((batch_size, max_sum_n_samples, t.shape[-1]))
                 for k, t in self.pu.sample(n=batch_size * max_sum_n_samples).items()}
            v_new = {k: t.expand((max_sum_n_samples,) + t.shape).transpose(0, 1)
                     for k, t in v.items()}

            logpv = 0
            for k in self.v:
                if k in do:
                    if do[k] != v[k]:
                        return float('-inf')
                else:
                    logpv += self.f[k](v_new, u, v_new[k])
            assert tuple(logpv.shape) == (batch_size, max_sum_n_samples), \
                (logpv.shape, batch_size, max_sum_n_samples)

            # compute weights (max_n_samples - m + 1,)
            ik = np.arange(K.max())
            ipk = T.tensor(np.where(ik < alpha, ik, alpha * 0.9 ** (alpha - ik)),
                           device=self.device_param.device)

            # compute SUMO given samples (batch_size, n)
            indices = np.pad(n_samples, [(0, 0), (1, 0)]).cumsum(axis=1).astype(int)
            assert (np.diff(indices) > 0).all()
            estimates = T.zeros(batch_size, n, device=self.device_param.device)
            for i in range(batch_size):
                for j, (s, e) in enumerate(zip(indices[i, :-1], indices[i, 1:])):
                    assert e - s == n_samples[i, j]
                    samples = logpv[i, s:e]
                    vals = (T.logcumsumexp(samples, dim=0)
                            - T.log(T.arange(len(samples), device=self.device_param.device) + 1))
                    vals = vals[m-1:]
                    estimates[i][j] = vals[0] + (T.diff(vals) * ipk[:len(vals)-1]).sum()

            # return empirical mean of SUMO estimates per sample (batch_size,)
            if return_biased:
                return (-estimates.mean(dim=1),
                        T.logsumexp(logpv.flatten(), dim=0) - T.log(T.tensor(logpv.numel())))
            else:
                return -estimates.mean(dim=1)
        finally:
            self.train(mode=mode)

    def nll_marg(self, v, n=1, m=10000, do={}, return_biased=False):
        assert not set(v.keys()).difference(self.v)
        assert not set(do.keys()).difference(self.v)

        marg_set = set(self.v).difference(v.keys()).difference(do.keys())
        marg_space = self.space(select=marg_set)
        pv = 0
        biased_pv = 0
        for marg_v in marg_space:
            v_joined = dict()
            v_joined.update(marg_v)
            v_joined.update(v)
            v_joined.update(do)
            #nll_all, biased_nll = self.nll(v_joined, n=n, m=m, do=do, return_biased=return_biased)
            nll_all = self.biased_nll(v_joined, n=m, do=do)
            biased_nll = 0
            pv += T.exp(-nll_all)
            if return_biased:
                biased_pv += T.exp(-biased_nll)
        if return_biased:
            return -T.log(pv), -T.log(biased_pv)
        return -T.log(pv)

if __name__ == "__main__":
    backdoor_cg = CausalGraph(["A", "B", "C", "D"], [("A", "B"), ("A", "C"), ("B", "D")])
    target_node, fn, sn, on = backdoor_cg.categorize_neighbors()
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
    num_samples = 100
    # Instantiate distributions
    scm = SCM(target_node, fn, sn, on)
    fn_size, sn_size, on_size, u_size, i_size = scm.size()
    print(fn_size, sn_size, on_size, u_size, i_size)
    f = {var: MLP(backdoor_cg, i_size, u_size) for var in fn if i_size > 0 and u_size > 0}
    # print(f)
    # Create an instance of the FF_NCM class
    # Create an instance of the NCM class
    ncm = NCM(backdoor_cg, v_size={var: 1 for var in backdoor_cg.v},
              u_size={var: 1 for var in backdoor_cg.sn},
              f={var: MLP(backdoor_cg, len(backdoor_cg.v), len(backdoor_cg.sn)) for var in backdoor_cg.fn if
                 len(backdoor_cg.v) > 0 and len(backdoor_cg.sn) > 0})
    # # Test the model by sampling from it
    samples = ncm.forward(n=10)

    # # Print the generated samples
    # for k, v in samples.items():
    #     print(f"{k}: {v}")
