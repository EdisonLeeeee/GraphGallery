#!/usr/bin/env python
# coding: utf-8

import graphgallery 
import tensorflow as tf
from graphgallery import functional as gf

graphgallery.set_memory_growth()

print("GraphGallery version: ", graphgallery.__version__)
print("TensorFlow version: ", tf.__version__)

'''
Load Datasets
- cora/citeseer/pubmed/dblp/polblogs/cora_ml, etc...
'''
from graphgallery.datasets import Planetoid, NPZDataset
data = NPZDataset('cora', root="~/GraphData/datasets/", transform=gf.Standardize(), verbose=False)
graph = data.graph
splits = data.split_nodes(random_state=15)

from graphgallery.gallery import GCN
trainer = GCN(graph, graph_transform="SVD", device="gpu", seed=123)
trainer.build()
history = trainer.train(splits.train_nodes, splits.val_nodes, verbose=1, epochs=100)
results = trainer.test(splits.test_nodes)
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
