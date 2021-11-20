#!/usr/bin/env python
# coding: utf-8

import torch
import graphgallery
from graphgallery.datasets import Planetoid
from graphgallery.gallery import callbacks

print("GraphGallery version: ", graphgallery.__version__)
print("PyTorch version: ", torch.__version__)

'''
Load Datasets
- cora/citeseer/pubmed
'''


data = Planetoid('cora', root="~/GraphData/datasets/", verbose=False)
graph = data.graph
splits = data.split_nodes()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

graphgallery.set_backend("torch")
from graphgallery.gallery.nodeclas import GWNN

trainer = GWNN(device=device, seed=123).setup_graph(graph, feat_transform=None).build()
cb = callbacks.ModelCheckpoint('model.pth', monitor='val_accuracy')
trainer.fit(splits.train_nodes, splits.val_nodes, verbose=1, callbacks=[cb])
results = trainer.evaluate(splits.test_nodes)
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
