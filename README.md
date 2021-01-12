<p align="center">
  <img width = "600" height = "300" src="https://github.com/EdisonLeeeee/GraphGallery/blob/master/imgs/graphgallery.svg" alt="logo"/>
  <br/>
</p>

<p align="center"><strong><em>TensorFLow</em> or <em>PyTorch</em>? Both!</strong></p>
<!-- <p align="center"><strong>A <em>gallery</em> of state-of-the-art Graph Neural Networks (GNNs) for TensorFlow and PyTorch</strong>.</p> -->

<p align=center>
  <a href="https://www.python.org/downloads/release/python-370/">
    <img src="https://img.shields.io/badge/Python->=3.7-3776AB?logo=python" alt="Python">
  </a>    
  <a href="https://github.com/tensorflow/tensorflow/releases/tag/v2.1.0">
    <img src="https://img.shields.io/badge/TensorFlow->=2.1.0-FF6F00?logo=tensorflow" alt="tensorflow">
  </a>      
  <a href="https://github.com/pytorch/pytorch">
    <img src="https://img.shields.io/badge/PyTorch->=1.4-FF6F00?logo=pytorch" alt="pytorch">
  </a>   
  <a href="https://pypi.org/project/graphgallery/">
    <img src="https://badge.fury.io/py/graphgallery.svg" alt="pypi">
  </a>       
  <a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/EdisonLeeeee/GraphGallery" alt="license">
  </a>       
</p>

# GraphGallery
GraphGallery is a gallery for benchmarking Graph Neural Networks (GNNs) with [TensorFlow 2.x](https://github.com/tensorflow/tensorflow) and [PyTorch](https://github.com/pytorch/pytorch) backend. Besides, [Pytorch Geometric (PyG)](https://github.com/rusty1s/pytorch_geometric) backend and [Deep Graph Library (DGL)](https://github.com/dmlc/dgl) backend now are available in GraphGallery.

# 💨 NEWS
We have integrated the adversarial parts in this project, examples please refer to [Graph Adversarial Learning examples](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning)

# 🚀 Installation
pip install -U graphgallery

GraphGallery has been tested on:
+ CUDA 10.1
+ TensorFlow 2.1~2.4, 2.1.2 is recommended.
+ PyTorch 1.4~1.7
+ Pytorch Geometric (PyG) 1.6.1
+ DGL 0.5.2, 0.5.3

# 🤖 Implementations
Please refer to the [examples](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/) directory.
(The examples are updating...)

# ⚡ Quick Start on GNNs
## Datasets
more details please refer to [GraphData](https://github.com/EdisonLeeeee/GraphData).
## Example of GCN
```python
from graphgallery.gallery import GCN

# initialize a GNN trainer
trainer = GCN(graph)
# process your inputs, such as converting to tensors
trainer.process()
# build your GCN trainer with default hyper-parameters
trainer.build()
# train your trainer. here splits.train_nodes and splits.val_nodes are numpy arrays
# verbose takes 0, 1, 2, 3, 4
history = trainer.train(splits.train_nodes, splits.val_nodes, verbose=1, epochs=100)
# test your trainer
# verbose takes 0, 1, 2
results = trainer.test(splits.nodes, verbose=1)
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
```
Other models in the gallery are the same.

## Using Other Backend
```python
>>> import graphgallery
>>> graphgallery.backend()
TensorFlow 2.1.2 Backend

>>> graphgallery.set_backend("pytorch")
PyTorch 1.6.0+cu101 Backend

# DGL PyTorch backend
>>> graphgallery.set_backend("dgl")

# DGL TensorFlow backend
>>> graphgallery.set_backend("dgl-tf")
```
But you codes don't even need to change!

# ❓ How to add your datasets
This is motivated by [gnn-benchmark](https://github.com/shchur/gnn-benchmark/)
```python
from graphgallery.data import Graph

# Load the adjacency matrix A, attribute matrix X and labels vector y
# A - scipy.sparse.csr_matrix of shape [num_nodes, num_nodes]
# X - scipy.sparse.csr_matrix or np.ndarray of shape [num_nodes, num_attrs]
# y - np.ndarray of shape [num_nodes]

mydataset = Graph(adj_matrix=A, node_attr=X, node_label=y)
# save dataset
mydataset.to_npz('path/to/mydataset.npz')
# load dataset
mydataset = Graph.from_npz('path/to/mydataset.npz')
```


# ⭐ Road Map
- [x] Add PyTorch trainers support
- [x] Add other frameworks (PyG and DGL) support
- [ ] Add more GNN trainers (TF and Torch backend)
- [ ] Support for more tasks, e.g., `graph Classification` and `link prediction`
- [x] Support for more types of graphs, e.g., Heterogeneous graph
- [ ] Add Docstrings and Documentation (Building)
- [ ] Comprehensive tutorials

# 😘 Acknowledgement
This project is motivated by [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric), [Tensorflow Geometric](https://github.com/CrawlScript/tf_geometric), [Stellargraph](https://github.com/stellargraph/stellargraph) and [DGL](https://github.com/dmlc/dgl), etc., and the original implementations of the authors, thanks for their excellent works!