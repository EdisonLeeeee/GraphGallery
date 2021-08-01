<p align="center">
  <img width = "700" height = "300" src="https://github.com/EdisonLeeeee/GraphGallery/blob/master/imgs/graphgallery.svg" alt="banner"/>
  <br/>
</p>
<p align="center"><strong><em>GraphGallery</em> is a <em>gallery</em>, not a <em>Library</em></strong></p>

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
git clone https://github.com/EdisonLeeeee/GraphGallery.git && cd GraphGallery
pip install -e . --verbose
```
# 🤖 Implementations
In detail, the following methods are currently implemented:

## Node Classification Task

<!-- 1 -->
<details>
<summary>
<b>ChebyNet</b> from <i>Michaël Defferrard et al</i>,
<a href="https://arxiv.org/abs/1606.09375"> 📝<i>Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (NeurIPS'16)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/ChebyNet.py"> [:octocat:TensorFLow]</a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/ChebyNet.py"> [🔥PyTorch] </a>

</details>

<!-- 2 -->

<details>
<summary>
<b>GCN</b> from <i>Thomas N. Kipf et al</i>,
<a href="https://arxiv.org/abs/1609.02907"> 📝<i>Semi-Supervised Classification with Graph Convolutional Networks (ICLR'17)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/GCN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GCN.py"> [🔥PyTorch] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyG/GCN.py"> [🔥PyG] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/DGL-PyTorch/GCN.py"> [🔥DGL-PyTorch] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/DGL-TensorFlow/GCN.py"> [:octocat:DGL-TensorFlow] </a>
</details>

<!-- 3 -->
<details>
<summary>
<b>GraphSAGE</b> from <i>William L. Hamilton et al</i>,
<a href="https://arxiv.org/abs/1706.02216"> 📝<i>Inductive Representation Learning on Large Graphs (NeurIPS'17)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/GraphSAGE.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GraphSAGE.py"> [🔥PyTorch] </a>
</details>

<!-- 4 -->
<details>
<summary>
<b>FastGCN</b> from <i>Jie Chen et al</i>,
<a href="https://arxiv.org/abs/1801.10247"> 📝<i>FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling (ICLR'18)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/FastGCN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/FastGCN.py"> [🔥PyTorch] </a>
</details>

<!-- 5 -->
<details>
<summary>
<b>LGCN</b> from <i>Hongyang Gao et al</i>,
<a href="https://arxiv.org/abs/1808.03965"> 📝<i>Large-Scale Learnable Graph Convolutional Networks (KDD'18)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/LGCN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/LGCN.py"> [🔥PyTorch] </a>
</details>

<!-- 6 -->
<details>
<summary>
<b>GAT</b> from <i>Petar Veličković et al</i>,
<a href="https://arxiv.org/abs/1710.10903"> 📝<i>Graph Attention Networks (ICLR'18)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/GAT.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GAT.py"> [🔥PyTorch] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyG/GAT.py"> [🔥PyG] </a>
</details>

<!-- 7 -->
<details>
<summary>
<b>SGC</b> from <i>Felix Wu et al</i>,
<a href="https://arxiv.org/abs/1902.07153"> 📝<i>Simplifying Graph Convolutional Networks (ICLR'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/SGC.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/SGC.py"> [🔥PyTorch] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyG/SGC.py"> [🔥PyG] </a>
</details>

<!-- 8 -->
<details>
<summary>
<b>GWNN</b> from <i>Bingbing Xu et al</i>,
<a href="https://arxiv.org/abs/1904.07785"> 📝<i>Graph Wavelet Neural Network (ICLR'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/GWNN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GWNN.py"> [🔥PyTorch] </a>
</details>

<!-- 9 -->
<details>
<summary>
<b>GMNN</b> from <i>Meng Qu et al</i>,
<a href="https://arxiv.org/abs/1905.06214"> 📝<i>Graph Attention Networks (ICLR'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/GMNN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GMNN.py"> [🔥PyTorch] </a>
</details>

<!-- 10 -->
<details>
<summary>
<b>ClusterGCN</b> from <i>Wei-Lin Chiang et al</i>,
<a href="https://arxiv.org/abs/1905.07953"> 📝<i>Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks (KDD'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/ClusterGCN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/ClusterGCN.py"> [🔥PyTorch] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyG/ClusterGCN.py"> [🔥PyG] </a>
</details>

<!-- 11 -->
<details>
<summary>
<b>DAGNN</b> from <i>Meng Liu et al</i>,
<a href="https://arxiv.org/abs/2007.09296"> 📝<i>Towards Deeper Graph Neural Networks (KDD'20)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/DAGNN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/DAGNN.py"> [🔥PyTorch] </a>
</details>

<!-- 12 -->
<details>
<summary>
<b>GDC</b> from <i>Johannes Klicpera et al</i>,
<a href="https://www.in.tum.de/daml/gdc/"> 📝<i>Diffusion Improves Graph Learning (NeurIPS'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/GCN-GDC.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GCN-GDC.py"> [🔥PyTorch] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyG/GCN-GDC.py"> [🔥PyG] </a>
</details>

<!-- 13 -->

<details>
<summary>
<b>TAGCN</b> from <i>Du et al</i>,
<a href="https://arxiv.org/abs/1710.10370"> 📝<i>Topology Adaptive Graph Convolutional Networks (arxiv'17)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/TAGCN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/TAGCN.py"> [🔥PyTorch] </a>
</details>

<!-- 14 -->

<details>
<summary>
<b>APPNP, PPNP</b> from <i>Johannes Klicpera et al</i>,
<a href="https://arxiv.org/abs/1810.05997"> 📝<i>Predict then Propagate: Graph Neural Networks meet Personalized PageRank (ICLR'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/APPNP.py"> [:octocat:TensorFLow(APPNP)] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/PPNP.py"> [:octocat:TensorFLow(PPNP)] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/APPNP.py"> [🔥PyTorch(APPNP)] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/PPNP.py"> [🔥PyTorch(PPNP)] </a>
</details>

<!-- 15 -->

<details>
<summary>
<b>PDN</b> from <i>Benedek Rozemberczki et al</i>,
<a href="https://arxiv.org/abs/2010.12878"> 📝<i>Pathfinder Discovery Networks for Neural Message Passing (ICLR'21)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyG/PDN.py"> [🔥PyG] </a>
</details>
</details>
</details>

<!-- 16 -->

<details>
<summary>
<b>SSGC</b> from <i>Zhu et al</i>,
<a href="https://openreview.net/forum?id=CYO5T-YjWZV"> 📝<i>Simple Spectral Graph Convolution (ICLR'21)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/SSGC.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/SSGC.py"> [🔥PyTorch] </a>
</details>

<!-- 17 -->

<details>
<summary>
<b>AGNN</b> from <i>Zhu et al</i>,
<a href="https://arxiv.org/abs/1609.02907"> 📝<i>Attention-based Graph Neural Network for semi-supervised learning (ICLR'18 openreview)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/AGNN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/AGNN.py"> [🔥PyTorch] </a>
</details>

<!-- 18 -->

<details>
<summary>
<b>ARMA</b> from <i>Bianchi et al.</i>,
<a href="https://arxiv.org/abs/1901.01343"> 📝<i>Graph Neural Networks with convolutional ARMA filters (Arxiv'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/ARMA.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/ARMA.py"> [🔥PyTorch] </a>
</details>


<!-- 19 -->

<details>
<summary>
<b>GMLP</b> from <i>Yang Hu et al.</i>,
<a href="https://arxiv.org/abs/2106.04051"> 📝<i>Graph-MLP: Node Classification without Message Passing in Graph (Arxiv'21)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GraphMLP.py"> [🔥PyTorch] </a>
</details>

### Defense models (for Graph Adversarial Learning)

#### Robust Optimization

<!-- 1 -->
<details>
<summary>
<b>RobustGCN</b> from <i>Petar Veličković et al</i>,
<a href="https://dl.acm.org/doi/10.1145/3292500.3330851"> 📝<i>Robust Graph Convolutional Networks Against Adversarial Attacks (KDD'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/RobustGCN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/RobustGCN.py"> [🔥PyTorch] </a>
</details>

<!-- 2 -->
<details>
<summary>
<b>SBVAT</b> from <i>Zhijie Deng et al</i>,
<a href="https://arxiv.org/abs/1902.09192"> 📝<i>Batch Virtual Adversarial Training for Graph Convolutional Networks (ICML'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/SBVAT.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/SBVAT.py"> [🔥PyTorch] </a>
</details>

<!-- 3 -->
<details>
<summary>
<b>OBVAT</b> from <i>Zhijie Deng et al</i>,
<a href="https://arxiv.org/abs/1902.09192"> 📝<i>Batch Virtual Adversarial Training for Graph Convolutional Networks (ICML'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/OBVAT.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/OBVAT.py"> [🔥PyTorch] </a>
</details>

<!-- 4 -->
<details>
<summary>
<b>SimPGCN</b> from <i>Wei Jin et al</i>,
<a href="https://arxiv.org/abs/2011.09643"> 📝<i>Node Similarity Preserving Graph Convolutional Networks (WSDM'21)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/SimPGCN.py"> [🔥PyTorch] </a>
</details>

<!-- 5 -->
<details>
<summary>
<b>GCN-VAT, GraphVAT</b> from <i>Fuli Feng et al</i>,
<a href="https://arxiv.org/abs/1902.08226"> 📝<i>Graph Adversarial Training: Dynamically Regularizing Based on Graph Structure (TKDE'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GCN-VAT.py"> [🔥GCN-VAT-PyTorch] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GraphVAT.py"> [🔥GraphVAT-PyTorch] </a>
</details>

<!-- 6 -->
<details>
<summary>
<b>LATGCN</b> from <i>Hongwei Jin et al</i>,
<a href="https://graphreason.github.io/papers/35.pdf"> 📝<i>Latent Adversarial Training of Graph Convolution Networks (ICML@LRGSD'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/LATGCN.py"> [🔥PyTorch] </a>
</details>

<!-- 7 -->
<details>
<summary>
<b>DGAT</b> from <i>Weibo Hu et al</i>,
<a href="https://link.springer.com/article/10.1007/s10489-021-02272-y"> 📝<i>Robust graph convolutional networks with directional graph adversarial training (Applied Intelligence'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/DGAT.py"> [🔥PyTorch] </a>
</details>

#### Graph Purification

The graph purification methods are universal for all models, just specify:

```python
graph_transform="purification_method"
```

so, here we only give the examples of `GCN` with purification methods, other models should work.

<!-- 1 -->
<details>
<summary>
<b>GCN-Jaccard</b> from <i>Huijun Wu et al</i>,
<a href="https://arxiv.org/abs/1903.01610"> 📝<i>Adversarial Examples on Graph Data: Deep Insights into Attack and Defense (IJCAI'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/GCN-Jaccard.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GCN-Jaccard.py"> [🔥PyTorch] </a>
</details>

<!-- 2 -->
<details>
<summary>
<b>GCN-SVD</b> from <i>Negin Entezari et al</i>,
<a href="https://dl.acm.org/doi/abs/10.1145/3336191.3371789"> 📝<i>All You Need Is Low (Rank): Defending Against Adversarial Attacks on Graphs (WSDM'20)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/GCN-SVD.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GCN-SVD.py"> [🔥PyTorch] </a>
</details>

## Embedding

<!-- 1 -->
<details>
<summary>
<b>Deepwalk</b> from <i>Bryan Perozzi et al</i>,
<a href="https://arxiv.org/abs/1403.6652"> 📝<i>DeepWalk: Online Learning of Social Representations (KDD'14)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/Common/Deepwalk.py"> [Example] </a>
</details>

<!-- 2 -->
<details>
<summary>
<b>Node2vec</b> from <i>Aditya Grover and Jure Leskovec</i>,
<a href="https://arxiv.org/abs/1607.00653"> 📝<i>node2vec: Scalable Feature Learning for Networks (KDD'16)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/Common/Node2vec.py"> [Example] </a>
</details>

# ⚡ Quick Start on GNNs
## Datasets
more details please refer to [GraphData](https://github.com/EdisonLeeeee/GraphData).
## Example of GCN (Node Classification Task)
It takes just a few lines of code.
```python
from graphgallery.gallery.nodeclas import GCN
trainer = GCN()
trainer.setup_graph(graph)
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
Please cite our [paper](https://www.computer.org/csdl/proceedings-article/icse-companion/2021/121900a013/1sET5DXNWJG) (and the respective papers of the methods used) if you use this code in your own work:
```bibtex
@inproceedings{li2021graphgallery,
author = {J. Li and K. Xu and L. Chen and Z. Zheng and X. Liu},
booktitle = {2021 IEEE/ACM 43rd International Conference on Software Engineering: Companion Proceedings (ICSE-Companion)},
title = {GraphGallery: A Platform for Fast Benchmarking and Easy Development of Graph Neural Networks Based Intelligent Software},
year = {2021},
pages = {13-16},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
}

```