import itertools
import matplotlib.pyplot as plt
import causal
import torch.nn.functional as F
import example_scm
import torch as T
import torch.nn as nn
import numpy as np
from utils import *
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

if __name__ == "__main__":
    bernoulli_distribution = BernoulliDistribution(['U_A1', 'U_A2', 'U_AB', 'U_AC', 'U_B', 'U_C', 'U_D'],
                                                   {'U_A1': 1, 'U_A2': 1, 'U_AB': 1, 'U_AC': 1, 'U_B': 1, 'U_C': 1,
                                                    'U_D': 1},
                                                   p=0.5,
                                                   seed=0)
    scm = example_scm.SCM(bernoulli_distribution)
    # Training parameters
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
    num_samples = 200
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
        # print(prob)
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
    plt.savefig('Loss over epochs.png')

    # Plot the estimated probability over epochs
    plt.figure()
    plt.plot(range(epochs), estimated_probs)
    plt.xlabel('Epoch')
    plt.ylabel('Estimated P(A=1|C=1)')
    plt.title('Estimated P(A=1|C=1) over epochs')
    plt.savefig('Estimated.png')

    plt.show()