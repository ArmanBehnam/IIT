# 1
# pip install neo4j
# pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
# pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
# pip install torch-geometric

import os
import pickle
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# set seed
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
DATA_PATH = "/home"

# I downloaded the MUTAG database.

from torch_geometric.datasets import TUDataset

dataset = TUDataset(root=DATA_PATH, name='MUTAG')
print("len:", len(dataset))
print("num_classes:", dataset.num_classes)
print("num_node_features:", dataset.num_node_features)

# I continued setting up the graph database driver to a "Blank Sandbox" database instance that I set up here.

from neo4j import GraphDatabase, basic_auth

driver = GraphDatabase.driver("bolt://18.213.150.193:7687", auth=basic_auth("neo4j", "leave-weathers-conspiracies"))


cypher_query = '''
UNWIND $items AS item
MERGE (a:Atom {element: item.atom_0_elmt, atom_id: item.atom_0, molecule_id: item.mol_id, mutagenic: item.mutagen})
MERGE (b:Atom {element: item.atom_1_elmt, atom_id: item.atom_1, molecule_id: item.mol_id, mutagenic: item.mutagen})
MERGE (a)-[:BONDED_TO {bond_type: item.bond}]-(b)
'''

edges = []
elements = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
bonds = ['AROMATIC', 'SINGLE', 'DOUBLE', 'TRIPLE']

def create_graph(tx, edges):
  tx.run(cypher_query, items=edges)

with driver.session() as session:
  for mol_id in range(len(dataset)):
    for bond_id in range(dataset[mol_id]['edge_index'].size()[1]):
      atom_0 = int(dataset[mol_id]['edge_index'][:, bond_id][0])
      atom_1 = int(dataset[mol_id]['edge_index'][:, bond_id][1])
      atom_0_elmt = elements[int(torch.argmax(dataset[mol_id]['x'][atom_0]))]
      atom_1_elmt = elements[int(torch.argmax(dataset[mol_id]['x'][atom_1]))]
      bond = bonds[int(torch.argmax(dataset[mol_id]['edge_attr'][bond_id]))]
      edges.append(
          {'mol_id': mol_id,
          'mutagen': int(dataset[mol_id]['y']),
          'atom_0': atom_0,
          'atom_1': atom_1,
          'atom_0_elmt': atom_0_elmt,
          'atom_1_elmt': atom_1_elmt,
          'bond': bond}
          )
      if len(edges) % 1000 == 0:
          session.write_transaction(create_graph, edges)
          edges = []
  session.write_transaction(create_graph, edges)

driver.close()

# I retrieved data about the atoms and their properties via the following query sent via a read transaction.

cypher_query = '''
MATCH (a:Atom)
RETURN a.molecule_id AS molecule,
a.atom_id AS atom_id,
a.mutagenic AS y,
a.element AS element
'''

with driver.session(database="neo4j") as session:
  results_0 = session.read_transaction(
    lambda tx: tx.run(cypher_query).data())

driver.close()