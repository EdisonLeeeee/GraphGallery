#!/usr/bin/env python
# coding: utf-8

import graphgallery

print("GraphGallery version: ", graphgallery.__version__)

'''
Load Datasets
- cora/citeseer/pubmed
'''
from graphgallery.datasets import Planetoid
data = Planetoid('cora', root="~/GraphData/datasets/", verbose=False)
graph = data.graph
splits = data.split_nodes()

from graphgallery.gallery.embedding import BANE
trainer = BANE()
trainer.fit(graph.adj_matrix)
# embedding = trainer.get_embedding()
results = trainer.evaluate_nodeclas(graph.label,
                                    splits.train_nodes,
                                    splits.test_nodes)

print(results)
