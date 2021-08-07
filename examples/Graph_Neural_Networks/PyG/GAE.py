#!/usr/bin/env python
# coding: utf-8

import torch
import graphgallery
import torch_geometric

print("GraphGallery version: ", graphgallery.__version__)
print("Torch version: ", torch.__version__)
print("Torch_Geometric version: ", torch_geometric.__version__)
'''
Load Datasets
- cora/citeseer/pubmed
'''
from graphgallery.datasets import Planetoid
data = Planetoid('cora', root="~/GraphData/datasets/", verbose=False)
graph = data.graph
splits = data.split_edges(random_state=15)

graphgallery.set_backend("pyg")

from graphgallery.gallery.linkpred import GAE
trainer = GAE(device="gpu", seed=123)
trainer.setup_graph(graph).build()
his = trainer.fit(splits.train_pos_edge_index, verbose=3, epochs=100)
results = trainer.evaluate((splits.test_pos_edge_index, splits.test_neg_edge_index))
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
