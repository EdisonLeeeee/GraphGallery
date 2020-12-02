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

from graphgallery.gallery import SBVAT
model = SBVAT(graph, attr_transform="normalize_attr", device="gpu", seed=123)
model.build()
his = model.train(splits.train_nodes, splits.val_nodes, verbose=1, epochs=200)
results = model.test(splits.test_nodes) 
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
