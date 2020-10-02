import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
import graphgallery

from graphgallery.data import Planetoid
data = Planetoid('cora', root="~/GraphData/datasets/", verbose=False)
graph = data.graph
idx_train, idx_val, idx_test = data.split()

print(graph)

from graphgallery.nn.models import EdgeGCN

model = EdgeGCN(graph, attr_transform="normalize_attr",
                device="CPU", seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=1, epochs=100)
loss, accuracy = model.test(idx_test)
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')

# for testing the predict method
print(f'Predict accuracy {model._test_predict(idx_test):.2%}')
