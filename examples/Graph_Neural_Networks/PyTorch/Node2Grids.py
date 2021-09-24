#!/usr/bin/env python
# coding: utf-8

import torch
import graphgallery

print("GraphGallery version: ", graphgallery.__version__)
print("Torch version: ", torch.__version__)

'''
Load Datasets
- cora/citeseer/pubmed
'''
from graphgallery.datasets import Planetoid
data = Planetoid('cora', root="~/GraphData/datasets/", verbose=False)
graph = data.graph
splits = data.split_nodes()

graphgallery.set_backend("pytorch")

from graphgallery.gallery.nodeclas import Node2Grids
trainer = Node2Grids(device="gpu", seed=123)

# for cora
trainer.custom_setup(batch_size_train=20)
trainer.setup_graph(graph, mapsize_a=8)
trainer.build(dropout=0.8, conv_channel=16, hids=64)
# for citeseer
# trainer.custom_setup(batch_size_train=8)
# trainer.setup_graph(graph, mapsize_a=12)
# trainer.build(dropout=0.8, conv_channel=64, hids=200)
# for pubmed
# trainer.custom_setup(batch_size_train=8)
# trainer.setup_graph(graph, mapsize_a=8)
# trainer.build()
his = trainer.fit(splits.train_nodes, splits.val_nodes, verbose=1, epochs=100)
results = trainer.evaluate(splits.test_nodes)
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
