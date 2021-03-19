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


from graphgallery.embedding import Node2Vec
model = Node2Vec()
model.fit(graph.adj_matrix)
embedding = model.get_embedding()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

x_train = embedding[splits.train_nodes]
x_test = embedding[splits.test_nodes]
y_train = graph.node_label[splits.train_nodes]
y_test = graph.node_label[splits.test_nodes]

clf = LogisticRegression(solver="lbfgs",
                         max_iter=1000,
                         multi_class='auto',
                         random_state=None)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Test accuracy {accuracy:.2%}')
