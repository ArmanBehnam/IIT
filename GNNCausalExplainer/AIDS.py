import math
import alg1
from alg3 import *
import causal
import pandas as pd
from causal import *
import torch
from itertools import product

AIDS = "D:/University/Spring2023/Research/Session23-230608/Codes/My/data/AIDS/"
AIDS_df = pd.read_csv(AIDS + 'AIDS_A.txt', sep=',', header=None, names=['from', 'to'])
AIDS_graph_indicator = pd.read_csv(AIDS + 'AIDS_graph_indicator.txt', header=None, names=['graph_id'])
AIDS_node_labels = pd.read_csv(AIDS + 'AIDS_node_labels.txt', header=None, names=['node_label'])
AIDS_df['graph_id'] = AIDS_graph_indicator
AIDS_df['node_label'] = AIDS_node_labels

grouped = AIDS_df.groupby('graph_id')
AIDS_causal_graphs = {}
for graph_id, group in grouped:
    V = set(group['from']).union(set(group['to']))
    edges = list(zip(group['from'], group['to'])) + list(zip(group['to'], group['from']))
    AIDS_causal_graphs[graph_id] = causal.CausalGraph(V=V, path=edges)

cg = AIDS_causal_graphs[1.0]
v_star, one_hop_neighbors, two_hop_neighbors, out_of_neighborhood = cg.categorize_neighbors(target_node=cg.sort()[0])
print(f"Target node: {v_star}")
print(f"1-hop neighbors of A: {one_hop_neighbors}")
print(f"2-hop neighbors of A: {two_hop_neighbors}")
print(f"Out of neighborhood of A: {out_of_neighborhood}")
cg.plot()

# hyperparameters
num_epochs = 2
learning_rates = [0.001, 0.002, 0.005,0.01]
hidden_sizes = [32, 64, 128, 256]
num_layers = [1, 2, 3, 4]
lambdas = [0.01, 0.05, .1,.2,.3]
hyperparameters = product(learning_rates, hidden_sizes, num_layers, lambdas)
total_loss = []
for i, hyperparams in enumerate(hyperparameters):
    learning_rate, h_size, h_layers, lambdas = hyperparams
    print(f'Training with learning rate: {learning_rate}, h_size: {h_size}, h_layers: {h_layers}, lambdas: {lambdas}','\n')
    causal_loss = alg1.train(cg, lambdas, learning_rate, h_size, h_layers, num_epochs)
    total_loss.append(causal_loss)
total_loss = [x for x in total_loss if not math.isnan(x[0])]
print(total_loss)

plt.figure()
plt.plot(total_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.savefig('AIDS NCM Loss over epochs.png')
plt.show()
