import math
import pandas as pd
from matplotlib import pyplot as plt
import causal
import torch
import alg1
import time

def GNNCausalExplanation(dataset, lambdas, learning_rate, h_size, h_layers, num_epochs, delta):
    total_loss = []
    for graph_id in dataset:
        cg = dataset[graph_id]
        v_star, one_hop_neighbors, two_hop_neighbors, out_of_neighborhood = cg.categorize_neighbors(target_node=cg.sort()[0])
        print(f"This is group: {graph_id} and this is first subgraph ")
        print(f"Target node: {v_star}")
        print(f"1-hop neighbors of A: {one_hop_neighbors}")
        print(f"2-hop neighbors of A: {two_hop_neighbors}")
        print(f"Out of neighborhood of A: {out_of_neighborhood}")
        causal_loss = alg1.train(cg, lambdas = lambdas, learning_rate = learning_rate, h_size = h_size, h_layers = h_layers, num_epochs = num_epochs)
        print("The loss value is : ",causal_loss,'\n')
        total_loss.append(causal_loss)
    total_loss = [x for x in total_loss if not math.isnan(x[0])]
    print(total_loss)

    plt.figure()
    plt.plot(total_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.savefig('alg3 NCM Loss over epochs.png')
    plt.show()

if __name__ == "__main__":
    start_time1 = time.time()
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

    Mutagenicity = "D:/University/Spring2023/Research/Session23-230608/Codes/My/data/Mutagenicity/"
    Mutagenicity_df = pd.read_csv(Mutagenicity + 'Mutagenicity_A.txt', sep=',', header=None, names=['from', 'to'])
    Mutagenicity_graph_indicator = pd.read_csv(Mutagenicity + 'Mutagenicity_graph_indicator.txt', header=None,names=['graph_id'])
    Mutagenicity_node_labels = pd.read_csv(Mutagenicity + 'Mutagenicity_node_labels.txt', header=None,names=['node_label'])
    Mutagenicity_df['graph_id'] = Mutagenicity_graph_indicator
    Mutagenicity_df['node_label'] = Mutagenicity_node_labels
    grouped = Mutagenicity_df.groupby('graph_id')
    Mutagenicity_causal_graphs = {}
    for graph_id, group in grouped:
        V = set(group['from']).union(set(group['to']))
        edges = list(zip(group['from'], group['to'])) + list(zip(group['to'], group['from']))
        Mutagenicity_causal_graphs[graph_id] = causal.CausalGraph(V=V, path=edges)
    end_time1 = time.time()
    t1 = end_time1 - start_time1
    print('The time of the data preprocessing was : ', t1)

    num_subgraph_limit = 5
    delta = 0.01
    num_nodes_density = 5
    start_time2 = time.time()
    GNNCausalExplanation(dataset= AIDS_causal_graphs, lambdas = 0.1, learning_rate = 0.001, h_size = 128, h_layers = 2, num_epochs = 2, delta=0.01)
    end_time2 = time.time()
    t2 = end_time2 - start_time2
    print('The time of the calculation was : ',t2)