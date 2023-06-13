import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Uniform, Gumbel
import causal
import matplotlib.pyplot as plt
from itertools import product


 def neural_identification(F, do_x, do_y):
    ID = []
    for model in F:
        # For each input feature in the model
        for name, param in model.named_parameters():
            if 'weight' in name:
                theta_min = torch.min(param.data)
                theta_max = torch.max(param.data)
                param.data.fill_(theta_min)
                output_min = model(do_x)
                param.data.fill_(theta_max)
                output_max = model(do_x)
                # If the output for minimum and maximum theta is the same, the feature is identified as causal
                if torch.allclose(output_min, output_max):
                    query = generate_query(do_x, do_y)  # Assume there is a method to generate a query
                    ID.append(query)
    Exp = calculate_exp(ID)  # Assume there is a method to calculate the expressivity
    return ID, Exp


# # Step 3 and 4: Get the 1-hop neighborhood and create the initial subgraph
# G = set(one_hop_neighbors) | {v_star}
# print(G)
# for n in range(num_subgraph_limit):
#
# # Step 5 and 6: Get the 2-hop neighborhood and add nodes to create the next subgraph
# G_n_plus_1 = G.copy()
#     for _ in range(num_nodes_density):
#         if two_hop_neighbors:
#             random_node = random.choice(list(two_hop_neighbors))
#             G_n_plus_1.add(random_node)
#             two_hop_neighbors.discard(random_node)
#             if G_n:
#                 G_n.remove(random.choice(list(G_n)))  # Remove a random node
#
#     # Step 7: Iterate over the limit
#     for n in range(NSL):
#         # Step 7.1 to 7.3: Use the original code for Algorithms 1 and 2 (not provided here)
#         # We need to calculate the models and expressivity for the current and previous subgraphs
#
#         # Step 7.4: Check the change in expressivity
#         Exp_G_n = self.calculate_probabilities(self.generate_binary_values(G_n,
#                                                                            1000))  # Calculating probabilities as a stand-in for expressivity calculation
#         Exp_G_n_plus_1 = self.calculate_probabilities(self.generate_binary_values(G_n_plus_1,
#                                                                                   1000))  # Calculating probabilities as a stand-in for expressivity calculation
#         if abs(sum(Exp_G_n_plus_1.values()) - sum(Exp_G_n.values())) < delta or n == NSL:
#             break
#
#         # Prepare for next iteration
#         G_n = G_n_plus_1.copy()
#         if two_hop_neighbors:
#             random_node = random.choice(list(two_hop_neighbors))
#             G_n_plus_1.add(random_node)
#             two_hop_neighbors.discard(random_node)
#             if G_n:
#                 G_n.remove(random.choice(list(G_n)))  # Remove a random node
#
#     # Step 8: Declare the final subgraph as explanatory
#     # I'm assuming here that we can use the total probability of nodes as a stand-in for model selection
#     M_best = \
#     max(self.calculate_probabilities(self.generate_binary_values(G_n_plus_1, 1000)).items(), key=lambda x: x[1])[
#         0]  # Selecting the node with highest probability as the best model
#     Gamma_GNN = (G_n_plus_1, M_best)
#     return Gamma_GNN

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


# Then use it as:
F = []
for i in range(10):
    ncm = NCM(cg, lambda_reg=0.1, learning_rate=0.01, h_size = 64, h_layers = 2)
    losses = ncm.train(num_epochs=3)
    F.append(ncm.model)

do_x = torch.randn(1, 10)  # Some input for do operation
do_y = torch.randn(1, 1)  # Some output for do operation
ID, Exp = neural_identification(F, do_x, do_y)
