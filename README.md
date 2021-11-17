<p align="center">
  <img width = "700" height = "300" src="https://github.com/EdisonLeeeee/GraphGallery/blob/master/imgs/graphgallery.svg" alt="banner"/>
  <br/>
</p>
<p align="center"><strong><em>PyTorch</em> is all you need!</strong></p>

<p align=center>
  <a href="https://www.python.org/downloads/release/python-360/">
    <img src="https://img.shields.io/badge/Python->=3.6-3776AB?logo=python" alt="Python">
  </a>    
  <!-- <a href="https://github.com/tensorflow/tensorflow/releases/tag/v2.1.0">
    <img src="https://img.shields.io/badge/TensorFlow->=2.1.0-FF6F00?logo=tensorflow" alt="tensorflow">
  </a>       -->
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
GraphGallery is a gallery for benchmarking Graph Neural Networks (GNNs) [PyTorch](https://github.com/pytorch/pytorch) backend. Besides, [Pytorch Geometric (PyG)](https://github.com/pyg-team/pytorch_geometric) and [Deep Graph Library (DGL)](https://www.dgl.ai/) backend now are also alternative choices in GraphGallery.


# 💨 NEWS
+ We now no longer support the TensorFlow backend.

# 🚀 Installation
Please make sure you have installed [PyTorch](https://pytorch.org/). Also, [Pytorch Geometric (PyG)](https://github.com/pyg-team/pytorch_geometric) and [Deep Graph Library (DGL)](https://www.dgl.ai/) are alternative choices.

```bash
# Maybe outdated
pip install -U graphgallery
```
or
```bash
# Recommended
git clone https://github.com/EdisonLeeeee/GraphGallery.git && cd GraphGallery
pip install -e . --verbose
```
where `-e` means "editable" mode so you don't have to reinstall every time you make changes.

# 🤖 Implementations
In detail, the following methods are currently implemented:

## Node Classification
| Method                     | Author                       | Paper                                                                                                                                                                     | PyTorch            | TensorFlow         | PyG                | DGL                |
| -------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| **ChebyNet**               | *Michaël Defferrard et al*   | [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (NeurIPS'16)](https://arxiv.org/abs/1606.09375)                                           | :heavy_check_mark: | :heavy_check_mark: |                    |                    |
| **GCN**                    | *Thomas N. Kipf et al*       | [Semi-Supervised Classification with Graph Convolutional Networks (ICLR'17)](https://arxiv.org/abs/1609.02907)                                                            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| **GraphSAGE**              | *William L. Hamilton et al*  | [Inductive Representation Learning on Large Graphs (NeurIPS'17)](https://arxiv.org/abs/1706.02216)                                                                        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| **FastGCN**                | *Jie Chen et al*             | [FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling (ICLR'18)](https://arxiv.org/abs/1801.10247)                                            | :heavy_check_mark: | :heavy_check_mark: |                    |                    |
| **LGCN**                   | *Hongyang Gao et al*         | [Large-Scale Learnable Graph Convolutional Networks (KDD'18)](https://arxiv.org/abs/1808.03965)                                                                           |                    | :heavy_check_mark: |                    |                    |
| **GAT**                    | *Petar Veličković et al*     | [Graph Attention Networks (ICLR'18)](https://arxiv.org/abs/1710.10903)                                                                                                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| **SGC**                    | *Felix Wu et al*             | [Simplifying Graph Convolutional Networks (ICLR'19)](https://arxiv.org/abs/1902.07153)                                                                                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| **GWNN**                   | *Bingbing Xu et al*          | [Graph Wavelet Neural Network (ICLR'19)](https://arxiv.org/abs/1904.07785)                                                                                                | :heavy_check_mark: | :heavy_check_mark: |                    |                    |
| **GMNN**                   | *Meng Qu et al*              | [Graph Attention Networks (ICLR'19)](https://arxiv.org/abs/1905.06214)                                                                                                    |                    | :heavy_check_mark: |                    |                    |
| **ClusterGCN**             | *Wei-Lin Chiang et al*       | [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks (KDD'19)](https://arxiv.org/abs/1905.07953)                                 | :heavy_check_mark: | :heavy_check_mark: |                    |                    |
| **DAGNN**                  | *Meng Liu et al*             | [Towards Deeper Graph Neural Networks (KDD'20)](https://arxiv.org/abs/2007.09296)                                                                                         | :heavy_check_mark: | :heavy_check_mark: |                    |                    |
| **GDC**                    | *Johannes Klicpera et al*    | [Diffusion Improves Graph Learning (NeurIPS'19)](https://www.in.tum.de/daml/gdc/)                                                                                         | :heavy_check_mark: | :heavy_check_mark: |                    |                    |
| **TAGCN**                  | *Jian Du et al*              | [Topology Adaptive Graph Convolutional Networks (arxiv'17)](https://arxiv.org/abs/1710.10370)                                                                             | :heavy_check_mark: | :heavy_check_mark: |                    |                    |
| **APPNP, PPNP**            | *Johannes Klicpera et al*    | [Predict then Propagate: Graph Neural Networks meet Personalized PageRank (ICLR'19)](https://arxiv.org/abs/1810.05997)                                                    | :heavy_check_mark: | :heavy_check_mark: |                    |                    |
| **PDN**                    | *Benedek Rozemberczki et al* | [Pathfinder Discovery Networks for Neural Message Passing (ICLR'21)](https://arxiv.org/abs/2010.12878)                                                                    |                    |                    | :heavy_check_mark: |                    |
| **SSGC**                   | *Zhu et al*                  | [Simple Spectral Graph Convolution (ICLR'21)](https://openreview.net/forum?id=CYO5T-YjWZV)                                                                                | :heavy_check_mark: | :heavy_check_mark: |                    |                    |
| **AGNN**                   | *Zhu et al*                  | [Attention-based Graph Neural Network for semi-supervised learning (ICLR'18 openreview)](https://arxiv.org/abs/1609.02907)                                                | :heavy_check_mark: | :heavy_check_mark: |                    |                    |
| **ARMA**                   | *Bianchi et al*              | [Graph Neural Networks with convolutional ARMA filters (Arxiv'19)](https://arxiv.org/abs/1901.01343)                                                                      |                    | :heavy_check_mark: |                    |                    |
| **GraphML*P***             | *Yang Hu et al*              | [Graph-MLP: Node Classification without Message Passing in Graph (Arxiv'21)](https://arxiv.org/abs/2106.04051)                                                            | :heavy_check_mark: |                    |                    |                    |
| **LGC, EGC, hLGC**         | *Luca Pasa et al*            | [Simple Graph Convolutional Networks (Arxiv'21)](https://arxiv.org/abs/2106.05809)                                                                                        |                    |                    |                    | :heavy_check_mark: |
| **GRAND**                  | *Wenzheng Feng et al*        | [Graph Random Neural Network for Semi-Supervised Learning on Graphs (NeurIPS'20)](https://arxiv.org/abs/2005.11079)                                                       |                    |                    |                    | :heavy_check_mark: |
| **AlaGCN, AlaGAT**         | *Yiqing Xie et al*           | [When Do GNNs Work: Understanding and Improving Neighborhood Aggregation (IJCAI'20)](https://www.ijcai.org/Proceedings/2020/0181.pdf)                                     |                    |                    |                    | :heavy_check_mark: |
| **JKNet**                  | *Keyulu Xu et al*            | [Representation Learning on Graphs with Jumping Knowledge Networks (ICML'18)](https://arxiv.org/abs/1806.03536)                                                           |                    |                    |                    | :heavy_check_mark: |
| **MixHop**                 | *Sami Abu-El-Haija et al*    | [MixHop: Higher-Order Graph Convolutional Architecturesvia Sparsified Neighborhood Mixing (ICML'19)](https://arxiv.org/abs/1905.00067)                                    |                    |                    |                    | :heavy_check_mark: |
| **DropEdge**               | *Yu Rong et al*              | [DropEdge: Towards Deep Graph Convolutional Networks on Node Classification (ICML'20)](https://arxiv.org/abs/1907.10903)                                                  |                    |                    | :heavy_check_mark: |                    |
| **Node2Grids**             | *Dalong Yang et al*          | [Node2Grids: A Cost-Efficient Uncoupled Training Framework for Large-Scale Graph Learning (CIKM'21)](https://arxiv.org/abs/2003.09638)                                    | :heavy_check_mark: |                    |                    |                    |
| **RobustGCN**              | *Petar Veličković et al*     | [Robust Graph Convolutional Networks Against Adversarial Attacks (KDD'19)](https://dl.acm.org/doi/10.1145/3292500.3330851)                                                | :heavy_check_mark: | :heavy_check_mark: |                    |                    |
| **SBVAT, OBVAT**           | *Zhijie Deng et al*          | [Batch Virtual Adversarial Training for Graph Convolutional Networks (ICML'19)](https://arxiv.org/abs/1902.09192)                                                         | :heavy_check_mark: | :heavy_check_mark: |                    |                    |
| **SimPGCN**                | *Wei Jin et al*              | [Node Similarity Preserving Graph Convolutional Networks (WSDM'21)](https://arxiv.org/abs/2011.09643)                                                                     | :heavy_check_mark: |                    |                    |                    |
| **GCN-VAT, GraphVAT**      | *Fuli Feng et al*            | [Graph Adversarial Training: Dynamically Regularizing Based on Graph Structure (TKDE'19)](https://arxiv.org/abs/1902.08226)                                               | :heavy_check_mark: |                    |                    |                    |
| **LATGCN**                 | *Hongwei Jin et al*          | [Latent Adversarial Training of Graph Convolution Networks (ICML@LRGSD'19)](https://graphreason.github.io/papers/35.pdf)                                                  | :heavy_check_mark: |                    |                    |                    |
| **DGAT**                   | *Weibo Hu et al*             | [Robust graph convolutional networks with directional graph adversarial training (Applied Intelligence'19)](https://link.springer.com/article/10.1007/s10489-021-02272-y) | :heavy_check_mark: |                    |                    |                    |
| **MedianGCN , TrimmedGCN** | *Liang Chen et al*           | [Understanding Structural Vulnerability in Graph Convolutional Networks](https://arxiv.org/abs/2108.06280)                                                                | :heavy_check_mark: |                    | :heavy_check_mark: |                    |

#### Graph Purification

The graph purification methods are universal for all models, just specify:

```python
graph_transform="purification_method"
```

so, here we only give the examples of `GCN` with purification methods, other models should work.

| Method          | Author                 | Paper                                                                                                                                       | PyTorch            | TensorFlow         | PyG                | DGL                |
| --------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| **GCN-Jaccard** | *Huijun Wu et al*      | [Adversarial Examples on Graph Data: Deep Insights into Attack and Defense (IJCAI'19)](https://arxiv.org/abs/1903.01610)                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| **GCN-SVD**     | *Negin Entezari et al* | [All You Need Is Low (Rank): Defending Against Adversarial Attacks on Graphs (WSDM'20)](https://dl.acm.org/doi/abs/10.1145/3336191.3371789) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

## LinkPrediction
| Method        | Author                 | Paper                                                                           | PyTorch            | TensorFlow | PyG                | DGL |
| ------------- | ---------------------- | ------------------------------------------------------------------------------- | ------------------ | ---------- | ------------------ | --- |
| **GAE, VGAE** | *Thomas N. Kipf et al* | [Variational Graph Auto-Encoders (NeuIPS'16)](https://arxiv.org/abs/1611.07308) | :heavy_check_mark: |            | :heavy_check_mark: |     |

## Node Embedding
The following methods are framework-agnostic.

| Method        | Author                            | Paper                                                                                                           | PyTorch            | TensorFlow         | PyG                | DGL                |
| ------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| **Deepwalk**  | *Bryan Perozzi et al*             | [DeepWalk: Online Learning of Social Representations (KDD'14)](https://arxiv.org/abs/1403.6652)                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| **Node2vec**  | *Aditya Grover and Jure Leskovec* | [node2vec: Scalable Feature Learning for Networks (KDD'16)](https://arxiv.org/abs/1607.00653)                   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| **Node2vec+** | *Renming Liu et al*               | [Accurately Modeling Biased Random Walks on Weighted Graphs Using  Node2vec+](https://arxiv.org/abs/2109.08031) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| **BANE**      | *Hong Yang et al*                 | [Binarized attributed network embedding (ICDM'18)](https://ieeexplore.ieee.org/document/8626170)                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |



# ⚡ Quick Start on GNNs
## Datasets
more details please refer to [GraphData](https://github.com/EdisonLeeeee/GraphData).
## Example of GCN (Node Classification Task)
It takes just a few lines of code.
```python
from graphgallery.gallery.nodeclas import GCN
trainer = GCN()
trainer.setup_graph(graph).build()
trainer.fit(train_nodes)
results = trainer.evaluate(test_nodes)
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
```
Other models in the gallery are the same.

If you have any troubles, you can simply run `trainer.help()` for more messages.

## Other Backends
```python
>>> import graphgallery
# Default: PyTorch backend
>>> graphgallery.backend()
PyTorch 1.9.0+cu111 Backend
# Switch to PyTorch Geometric backend
>>> graphgallery.set_backend("pyg")
# Switch to DGL PyTorch backend
>>> graphgallery.set_backend("dgl")
# Switch to PyTorch backend
>>> graphgallery.set_backend("th")
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
- [x] set tensorflow as optional dependency when using graphgallery
- [ ] Add more GNN trainers (TF and Torch backend)
- [ ] Support for more tasks, e.g., `graph Classification` and `link prediction`
- [x] Support for more types of graphs, e.g., Heterogeneous graph
- [ ] Add Docstrings and Documentation (Building)
- [ ] Comprehensive tutorials

# ❓ FAQ

Please fell free to contact me if you have any troubles.

# 😘 Acknowledgement
This project is motivated by [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric), [Stellargraph](https://github.com/stellargraph/stellargraph) and [DGL](https://www.dgl.ai/), etc., and the original implementations of the authors, thanks for their excellent works!

# Cite
Please cite our [paper](https://www.computer.org/csdl/proceedings-article/icse-companion/2021/121900a013/1sET5DXNWJG) (and the respective papers of the methods used) if you use this code in your own work:
```bibtex
@inproceedings{li2021graphgallery,
author = {Jintang Li and Kun Xu and Liang Chen and Zibin Zheng and Xiao Liu},
booktitle = {2021 IEEE/ACM 43rd International Conference on Software Engineering: Companion Proceedings (ICSE-Companion)},
title = {GraphGallery: A Platform for Fast Benchmarking and Easy Development of Graph Neural Networks Based Intelligent Software},
year = {2021},
pages = {13-16},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
}

```