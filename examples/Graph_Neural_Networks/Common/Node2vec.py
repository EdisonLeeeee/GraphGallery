#!/usr/bin/env python
# coding: utf-8

import graphgallery
import tensorflow as tf

graphgallery.set_memory_growth()

print("GraphGallery version: ", graphgallery.__version__)
print("TensorFlow version: ", tf.__version__)

'''
Load Datasets
- cora/citeseer/pubmed
'''
from graphgallery.datasets import Planetoid
data = Planetoid('cora', root="~/GraphData/datasets/", verbose=False)
graph = data.graph
splits = data.split_nodes()

from graphgallery.gallery import Node2vec
trainer = Node2vec(graph).process().build()
his = trainer.train(splits.train_nodes)
results = trainer.test(splits.test_nodes)
print(f'Test accuracy {results.accuracy:.2%}')
