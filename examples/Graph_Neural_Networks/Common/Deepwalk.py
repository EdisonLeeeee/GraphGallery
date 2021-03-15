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

from graphgallery.gallery.nodeclas import Deepwalk
trainer = Deepwalk(graph).process().build()
his = trainer.fit(splits.train_nodes)
results = trainer.evaluate(splits.test_nodes)
print(f'Test accuracy {results.accuracy:.2%}')
