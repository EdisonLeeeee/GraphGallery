#!/usr/bin/env python
# coding: utf-8

import torch
import graphgallery
import torch_geometric
from graphgallery.gallery import callbacks
from graphgallery.datasets import Planetoid

print("GraphGallery version: ", graphgallery.__version__)
print("PyTorch version: ", torch.__version__)
print("Torch_Geometric version: ", torch_geometric.__version__)
'''
Load Datasets
- cora/citeseer/pubmed
'''
data = Planetoid('cora', root="~/GraphData/datasets/", verbose=False)
graph = data.graph
splits = data.split_edges(random_state=15)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

graphgallery.set_backend("pyg")

from graphgallery.gallery.linkpred import GAE, VGAE
trainer = GAE(device=device, seed=123).setup_graph(graph).build()
# trainer = VGAE(device=device, seed=123).setup_graph(graph).build()
cb = callbacks.ModelCheckpoint('model.pth', monitor='val_ap')
trainer.fit(splits.train_pos_edge_index,
            val_data=(splits.val_pos_edge_index, splits.val_neg_edge_index), verbose=1, callbacks=[cb])
results = trainer.evaluate((splits.test_pos_edge_index, splits.test_neg_edge_index))
print(results)
