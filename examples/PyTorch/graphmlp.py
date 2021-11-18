#!/usr/bin/env python
# coding: utf-8

import torch
import graphgallery
from graphgallery.datasets import Planetoid
from graphgallery.gallery import callbacks

print("GraphGallery version: ", graphgallery.__version__)
print("Torch version: ", torch.__version__)

'''
Load Datasets
- cora/citeseer/pubmed
'''
data = Planetoid('cora', root="~/GraphData/datasets/", verbose=False)
graph = data.graph
splits = data.split_nodes()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

graphgallery.set_backend("pytorch")
from graphgallery.gallery.nodeclas import GraphMLP

# Cora
trainer = GraphMLP(device=device, seed=123, tau=1, alpha=10.0).setup_graph(graph, feat_transform="normalize_feat").build()
# Citeseer
# trainer = GraphMLP(device=device, seed=42, alpha=1.0, tau=0.5).setup_graph(graph, feat_transform="normalize_feat").build()
# # Pubmed
# trainer = GraphMLP(device=device, seed=123, tau=1, alpha=100, lr=0.1).setup_graph(graph, feat_transform="normalize_feat").build()

cb = callbacks.ModelCheckpoint('model.pth', monitor='val_accuracy')
trainer.fit(splits.train_nodes, splits.val_nodes, verbose=1, epochs=400, callbacks=[cb])
results = trainer.evaluate(splits.test_nodes)
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
