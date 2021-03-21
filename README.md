<p align="center">
  <img width = "400" height = "200" src="https://github.com/EdisonLeeeee/GraphGallery/blob/master/imgs/graphgallery.svg" alt="banner"/>
  <br/>
</p>

<p align="center"><strong><em>TensorFLow</em> or <em>PyTorch</em>? Both!</strong></p>
<!-- <p align="center"><strong>A <em>gallery</em> of state-of-the-art Graph Neural Networks (GNNs) for TensorFlow and PyTorch</strong>.</p> -->

<p align=center>
  <a href="https://www.python.org/downloads/release/python-360/">
    <img src="https://img.shields.io/badge/Python->=3.6-3776AB?logo=python" alt="Python">
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
GraphGallery is a gallery for benchmarking Graph Neural Networks (GNNs) and Graph Adversarial Learning with [TensorFlow 2.x](https://github.com/tensorflow/tensorflow) and [PyTorch](https://github.com/pytorch/pytorch) backend. Besides, [Pytorch Geometric (PyG)](https://github.com/rusty1s/pytorch_geometric) backend and [Deep Graph Library (DGL)](https://github.com/dmlc/dgl) backend now are available in GraphGallery.

# 💨 NEWS
We have integrated the **Adversarial Attacks** in this project, examples please refer to [Graph Adversarial Learning examples](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning).

# 🚀 Installation
```bash
pip install -U graphgallery
```
or
```bash
https://github.com/EdisonLeeeee/GraphGallery.git
cd GraphGallery
pip install -e .
```
GraphGallery has been tested on:
+ CPU, CUDA 10.1, CUDA 11.0
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
## Example of GCN (Node Classification Task)
It takes just a few lines of code.
```python
from graphgallery.gallery.nodeclas import GCN
trainer = GCN()
trainer.make_data(graph)
trainer.build()
history = trainer.fit(train_nodes, val_nodes)
results = trainer.evaluate(test_nodes)
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
```
Other models in the gallery are the same.

## Other Backends
```python
>>> import graphgallery
# Default: TensorFlow backend
>>> graphgallery.backend()
TensorFlow 2.1.2 Backend
# Switch to PyTorch backend
>>> graphgallery.set_backend("torch")
# Switch to TensorFlow backend
>>> graphgallery.set_backend("tf")
# Switch to PyTorch Geometric backend
>>> graphgallery.set_backend("pyg")
# Switch to DGL PyTorch backend
>>> graphgallery.set_backend("dgl")
# Switch to DGL TensorFlow backend
>>> graphgallery.set_backend("dgl-tf")
```
But your codes don't even need to change.

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

# Cite
Please cite our [paper](https://arxiv.org/abs/2102.07933) (and the respective papers of the methods used) if you use this code in your own work:
```bibtex
@article{li2021graphgallery,
  title={GraphGallery: A Platform for Fast Benchmarking and Easy Development of Graph Neural Networks Based Intelligent Software},
  author={Li, Jintang and Xu, Kun and Chen, Liang and Zheng, Zibin and Liu, Xiao},
  journal={arXiv preprint arXiv:2102.07933},
  year={2021}
}
```