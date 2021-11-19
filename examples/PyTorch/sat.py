#!/usr/bin/env python
# coding: utf-8

import torch
import graphgallery
from graphgallery.datasets import NPZDataset
from graphgallery.gallery import callbacks

print("GraphGallery version: ", graphgallery.__version__)
print("PyTorch version: ", torch.__version__)

'''
Load Datasets
- cora/citeseer/pubmed
'''


data = NPZDataset('cora', root="~/GraphData/datasets/", verbose=False, transform='standardize')
graph = data.graph
splits = data.split_nodes(random_state=15)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

graphgallery.set_backend("pytorch")
from graphgallery.gallery.nodeclas import SATGCN, SATSSGC, SATSGC


trainer = SATGCN(device=device, seed=123).setup_graph(graph).build()
# trainer = SATSGC(device=device, seed=123).setup_graph(graph).build()
# trainer = SATSSGC(device=device, seed=123).setup_graph(graph).build()
cb = callbacks.ModelCheckpoint('model.pth', monitor='val_accuracy')
trainer.fit(splits.train_nodes, splits.val_nodes, verbose=1, callbacks=[cb])
results = trainer.evaluate(splits.test_nodes)
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
