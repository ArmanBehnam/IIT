import numpy as np
import torch
from mlp1 import *

import causal1
import torch.nn.functional as F
from scm1 import SCM
import torch as T
import torch.nn as nn
import numpy as np


def log(x):
    return T.log(x + 1e-8)


def expand_do(val, n):
    if T.is_tensor(val):
        return T.tile(val, (n, 1))
    else:
        return T.unsqueeze(T.ones(n, dtype=float) * val, 1)


def check_equal(input, val):
    if T.is_tensor(val):
        return T.all(T.eq(input, T.tile(val, (input.shape[0], 1))), dim=1).bool()
    else:
        return T.squeeze(input == val)


def soft_equals(input, val):
    if T.is_tensor(val):
        return T.sum(T.abs(T.tile(val, (input.shape[0], 1)) - input), dim=1)
    else:
        return T.squeeze(T.abs(val - input))


def cross_entropy_compare(input, val):
    if T.is_tensor(val):
        raise NotImplementedError()
    else:
        if val == 1:
            return T.sum(-log(input))
        elif val == 0:
            return T.sum(-log(1 - input))
        else:
            raise ValueError("Comparison to {} of type {} is not allowed.".format(val, type(val)))

def log(x):
    return T.log(x + 1e-8)


def expand_do(val, n):
    if T.is_tensor(val):
        return T.tile(val, (n, 1))
    else:
        return T.unsqueeze(T.ones(n, dtype=float) * val, 1)


def check_equal(input, val):
    if T.is_tensor(val):
        return T.all(T.eq(input, T.tile(val, (input.shape[0], 1))), dim=1).bool()
    else:
        return T.squeeze(input == val)


def soft_equals(input, val):
    if T.is_tensor(val):
        return T.sum(T.abs(T.tile(val, (input.shape[0], 1)) - input), dim=1)
    else:
        return T.squeeze(T.abs(val - input))


def cross_entropy_compare(input, val):
    if T.is_tensor(val):
        raise NotImplementedError()
    else:
        if val == 1:
            return T.sum(-log(input))
        elif val == 0:
            return T.sum(-log(1 - input))
        else:
            raise ValueError("Comparison to {} of type {} is not allowed.".format(val, type(val)))

class XORModel(SCM):
    def __init__(self, cg, dim=1, p=0.5, seed=None):
        self.cg = cg
        self.dim = dim
        self.p = p

        sizes = dict()
        for V in cg.v:
            if V == 'X' or V == 'Y':
                sizes[V] = 1
            else:
                sizes[V] = dim

        self.confounders = {V: [] for V in self.cg.v}
        for V1, V2 in cg.ue:
            conf_name = "U_{}{}".format(V1, V2)
            self.confounders[V1].append(conf_name)
            self.confounders[V2].append(conf_name)
            sizes[conf_name] = 1

        super().__init__(
            f={V: self.get_xor_func(V) for V in cg},
            pu=BernoulliDistribution(list(sizes.keys()), sizes, p=p, seed=seed))

    def get_xor_func(self, V):
        conf_list = self.confounders[V]
        par_list = self.cg.fn[V]

        def xor_func(v, u):
            values = u[V]

            for conf in conf_list:
                values = torch.bitwise_xor(values, u[conf])
            for par in par_list:
                par_samp = v[par].long()
                if values.shape[1] >= par_samp.shape[1]:
                    values = torch.bitwise_xor(values, par_samp)
                else:
                    par_samp = torch.unsqueeze(torch.remainder(torch.sum(par_samp, 1), 2), 1)
                    values = torch.bitwise_xor(values, par_samp)

            return values

        return xor_func