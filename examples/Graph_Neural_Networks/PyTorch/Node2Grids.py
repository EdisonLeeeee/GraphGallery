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
from graphgallery.gallery.callbacks import EarlyStopping

trainer = Node2Grids(device="gpu", seed=123)
trainer.custom_setup(batch_size_train=10)
trainer.setup_graph(graph)
trainer.build()
cb = EarlyStopping(monitor='val_accuracy', verbose=1, patience=200)
trainer.fit(splits.train_nodes, splits.val_nodes, callbacks=[cb], verbose=1, epochs=1000)
results = trainer.evaluate(splits.test_nodes)
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
