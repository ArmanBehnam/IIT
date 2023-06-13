import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Uniform, Gumbel, Bernoulli
import matplotlib.pyplot as plt
from itertools import product
import os
import numpy as np

class NNModel(nn.Module):
    def __init__(self, u, input_size, output_size, h_size, h_layers):
        super(NNModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.h_size = h_size
        self.h_layers = h_layers
        self.u = u
        layers = [nn.Linear(self.input_size, self.h_size)]
        for l in range(h_layers - 1):
            layers.append(nn.Linear(self.h_size, self.h_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.h_size, self.output_size))
        self.nn = nn.Sequential(*layers)
        self.nn.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('relu'))

    def nn_forward(self):
        out = self.nn(self.u)
        return torch.sigmoid(out)

class NCM:
    def __init__(self, graph, target_node, lambda_reg, learning_rate,h_size, h_layers):
        self.graph = graph
        self.h_size = h_size
        self.h_layers = h_layers
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.target_node = target_node
        self.states = {graph.target_node: Bernoulli(0.5).sample((1,))}
        self.u_i = {v: Uniform(0, 1).sample((1,)) for v in graph.one_hop_neighbors | graph.two_hop_neighbors}
        self.u_ij = {v: Uniform(0, 1).sample((1,)) for v in graph.one_hop_neighbors}
        self.u_do = {v: Uniform(0, 1).sample((1,)) for v in graph.one_hop_neighbors}
        self.u = torch.cat(list(self.states.values()) + list(self.u_i.values()) + list(self.u_ij.values()), dim=0)
        self.model = NNModel(u = self.u, input_size=len(self.u), output_size=1,h_size = self.h_size, h_layers = self.h_layers)

    def ncm_forward(self):
        G_i = Gumbel(loc=torch.tensor([0.0]), scale=torch.tensor([1.0])).sample((1,))
        f = self.model.nn_forward()
        # print(self.states)
        if len(self.u_ij) > 0:
            return G_i + torch.log(f)
        else:
            return G_i + torch.log(1 - f)

def train(cg,lambdas,learning_rate,h_size,h_layers,num_epochs):
    dir_path = "./model"
    os.makedirs(dir_path, exist_ok=True)
    for i in range(num_epochs):
        total_loss = 0.0
        ncm_models = {}  # This will store each NCM model
        # print('epoch : ', i)
        sum_f = 0
        for node in cg.set_v:
            # Instantiate NCM for each node inside the loop
            cg.target_node, cg.one_hop_neighbors, cg.two_hop_neighbors, cg.out_of_neighborhood = cg.categorize_neighbors(target_node=node)
            ncm = NCM(cg, cg.target_node,lambda_reg=lambdas, learning_rate=learning_rate, h_size=h_size, h_layers=h_layers)
            ncm_models[node] = ncm  # Save the NCM model with the node as the key
            f = ncm.ncm_forward()
            torch.save(ncm.model.state_dict(), os.path.join(dir_path, f'model_node{node}_epoch{i}.pth'))
            sum_f += torch.sum(torch.abs(f))
        # print('The sum_f is : ',sum_f)
        p_L1 = sum_f / len(cg.set_v)
        p_L2 = torch.abs(torch.prod(p_L1) / len(cg.set_v))
        optimizer = optim.Adam(ncm.model.parameters(), lr=ncm.learning_rate)
        optimizer.zero_grad()
        loss = ((1 / len(cg.set_v)) * -torch.log(torch.sum(p_L1))) - (lambdas * torch.log(torch.sum(p_L2)))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # total_loss = [x for x in total_loss if not math.isnan(x[0])]
    print("The loss value is : ", total_loss, '\n')
    best_ncm_model = ncm_models[node]
    return total_loss, best_ncm_model, p_L2

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

if __name__ == "__main__":
    cg = CausalGraph(['A', 'B', 'C', 'D'], [('A', 'B'), ('A', 'C'), ('B', 'D')])
    cg.target_node, cg.one_hop_neighbors, cg.two_hop_neighbors, cg.out_of_neighborhood = cg.categorize_neighbors(target_node=cg.sort()[0])
    datasets = cg.generate_binary_values(cg, num_samples=10)
    p_v = cg.calculate_probabilities(datasets)
    p_v_joint = cg.calculate_joint_probabilities(datasets)
    print(datasets,'\n',p_v,'\n',p_v_joint)

    # hyperparameters
    num_epochs = 3
    learning_rates = [0.001, 0.002, 0.005,0.01]
    hidden_sizes = [32, 64, 128, 256]
    num_layers = [1, 2, 3, 4]
    lambdas = [0.01, 0.05, .1,.2,.3]
    hyperparameters = product(learning_rates, hidden_sizes, num_layers, lambdas)
    total_loss = []
    for i, hyperparams in enumerate(hyperparameters):
        learning_rate, h_size, h_layers, lambdas = hyperparams
        print(f'Training with learning rate: {learning_rate}, h_size: {h_size}, h_layers: {h_layers}, lambdas: {lambdas}')
        causal_loss,best_ncm_model,p_do = train(cg, lambdas, learning_rate, h_size, h_layers, num_epochs)
        total_loss.append(causal_loss)
    # total_loss = [x for x in total_loss if not math.isnan(x[0])]
    print(total_loss, best_ncm_model,p_do)

    plt.figure()
    plt.plot(total_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.savefig('syn Loss over epochs.png')
