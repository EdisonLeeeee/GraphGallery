#!/usr/bin/env python
# coding: utf-8


import torch
import graphgallery
from graphgallery import functional as gf

print("GraphGallery version: ", graphgallery.__version__)
print("Torch version: ", torch.__version__)

'''
Load Datasets
- cora/citeseer/pubmed/dblp/polblogs/cora_ml, etc...
'''
from graphgallery.datasets import Planetoid, NPZDataset
data = NPZDataset('cora', root="~/GraphData/datasets/", transform=gf.Standardize(), verbose=False)
graph = data.graph
splits = data.split_nodes(random_state=15)

graphgallery.set_backend("pytorch")

from graphgallery.gallery.nodeclas import GCN
trainer = GCN(device="gpu", seed=123).setup_graph(graph, graph_transform="SVD").build()
history = trainer.fit(splits.train_nodes, splits.val_nodes, verbose=1, epochs=100)
results = trainer.evaluate(splits.test_nodes)
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
