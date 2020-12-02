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
    <img src="https://img.shields.io/badge/TensorFlow->=2.1.2-FF6F00?logo=tensorflow" alt="tensorflow">
  </a>      
  <a href="https://github.com/pytorch/pytorch">
    <img src="https://img.shields.io/badge/PyTorch->=1.5-FF6F00?logo=pytorch" alt="pytorch">
  </a>   
  <a href="https://pypi.org/project/graphgallery/">
    <img src="https://badge.fury.io/py/graphgallery.svg" alt="pypi">
  </a>       
  <a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/EdisonLeeeee/GraphGallery" alt="pypi">
  </a>       
</p>

- [GraphGallery](#graphgallery)
- [👀 What's important](#-whats-important)
- [🚀 Installation](#-installation)
- [🤖 Implementations](#-implementations)
  - [Semi-supervised models](#semi-supervised-models)
    - [General models](#general-models)
    - [Defense models](#defense-models)
  - [Unsupervised models](#unsupervised-models)
- [⚡ Quick Start](#-quick-start)
  - [Datasets](#datasets)
    - [Planetoid](#planetoid)
    - [NPZDataset](#npzdataset)
  - [Tensor](#tensor)
  - [Example of GCN model](#example-of-gcn-model)
  - [Customization](#customization)
  - [Visualization](#visualization)
  - [Using TensorFlow/PyTorch Backend](#using-tensorflowpytorch-backend)
    - [GCN using PyTorch backend](#gcn-using-pytorch-backend)
- [❓ How to add your datasets](#-how-to-add-your-datasets)
- [❓ How to define your models](#-how-to-define-your-models)
    - [GCN using PyG backend](#gcn-using-pyg-backend)
- [😎 More Examples](#-more-examples)
- [⭐ Road Map](#-road-map)
- [😘 Acknowledgement](#-acknowledgement)

# GraphGallery
GraphGallery is a gallery for benchmarking Graph Neural Networks (GNNs) with [TensorFlow 2.x](https://github.com/tensorflow/tensorflow) and [PyTorch](https://github.com/pytorch/pytorch) backend. GraphGallery 0.6.x is a total re-write from previous versions, and some things have changed. 

NEWS: [PyG](https://github.com/rusty1s/pytorch_geometric) backend and [DGL](https://github.com/dmlc/dgl) backend now are supported in GraphGallery!


# 👀 What's important
Differences between GraphGallery and [Pytorch Geometric (PyG)](https://github.com/rusty1s/pytorch_geometric), [Deep Graph Library (DGL)](https://github.com/dmlc/dgl), etc...
+ PyG and DGL are just like **TensorFlow** while GraphGallery is more like **Keras**
+ GraphGallery is a plug-and-play and user-friendly toolbox
+ GraphGallery has high scalaribility for researchers to use


# 🚀 Installation
+ Build from source (latest version)
```bash
git clone https://github.com/EdisonLeeeee/GraphGallery.git
cd GraphGallery
python setup.py install
```
+ Or using pip (stable version)
```bash
pip install -U graphgallery
```
# 🤖 Implementations
In detail, the following methods are currently implemented:

## Semi-supervised models
### General models

<!-- 1 -->
<details>
<summary>
<b>ChebyNet</b> from <i>Michaël Defferrard et al</i>,
<a href="https://arxiv.org/abs/1606.09375"> 📝<i>Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (NeurIPS'16)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/ChebyNet.py"> :octocat:TensorFLow Example</a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/ChebyNet.py"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/ChebyNet.py"> [🔥PyG Example] </a>

</details>

<!-- 2 -->

<details>
<summary>
<b>GCN</b> from <i>Thomas N. Kipf et al</i>,
<a href="https://arxiv.org/abs/1609.02907"> 📝<i>Semi-Supervised Classification with Graph Convolutional Networks (ICLR'17)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/GCN.py"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/GCN.py"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/GCN.py"> [🔥PyG Example] </a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/DGL-PyTorch/GCN.py"> [🔥DGL-PyTorch Example] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/DGL-TensorFlow/GCN.py"> [:octocat:DGL-TensorFlow Example] </a>
</details>

<!-- 3 -->
<details>
<summary>
<b>GraphSAGE</b> from <i>William L. Hamilton et al</i>,
<a href="https://arxiv.org/abs/1706.02216"> 📝<i>Inductive Representation Learning on Large Graphs (NeurIPS'17)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/GraphSAGE.py"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/GraphSAGE.py"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/GraphSAGE.py"> [🔥PyG Example] </a>
</details>

<!-- 4 -->
<details>
<summary>
<b>FastGCN</b> from <i>Jie Chen et al</i>,
<a href="https://arxiv.org/abs/1801.10247"> 📝<i>FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling (ICLR'18)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/FastGCN.py"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/FastGCN.py"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/FastGCN.py"> [🔥PyG Example] </a>
</details>

<!-- 5 -->
<details>
<summary>
<b>LGCN</b> from <i>Hongyang Gao et al</i>,
<a href="https://arxiv.org/abs/1808.03965"> 📝<i>Large-Scale Learnable Graph Convolutional Networks (KDD'18)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/LGCN.py"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/LGCN.py"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/LGCN.py"> [🔥PyG Example] </a>
</details>

<!-- 6 -->
<details>
<summary>
<b>GAT</b> from <i>Petar Veličković et al</i>,
<a href="https://arxiv.org/abs/1710.10903"> 📝<i>Graph Attention Networks (ICLR'18)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/GAT.py"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/GAT.py"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/GAT.py"> [🔥PyG Example] </a>
</details>

<!-- 7 -->
<details>
<summary>
<b>SGC</b> from <i>Felix Wu et al</i>,
<a href="https://arxiv.org/abs/1902.07153"> 📝<i>Simplifying Graph Convolutional Networks (ICLR'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/SGC.py"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/SGC.py"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/SGC.py"> [🔥PyG Example] </a>
</details>

<!-- 8 -->
<details>
<summary>
<b>GWNN</b> from <i>Bingbing Xu et al</i>,
<a href="https://arxiv.org/abs/1904.07785"> 📝<i>Graph Wavelet Neural Network (ICLR'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/GWNN.py"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/GWNN.py"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/GWNN.py"> [🔥PyG Example] </a>
</details>

<!-- 69 -->
<details>
<summary>
<b>GMNN</b> from <i>Meng Qu et al</i>,
<a href="https://arxiv.org/abs/1905.06214"> 📝<i>Graph Attention Networks (ICLR'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/GMNN.py"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/GMNN.py"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/GMNN.py"> [🔥PyG Example] </a>
</details>

<!-- 10 -->
<details>
<summary>
<b>ClusterGCN</b> from <i>Wei-Lin Chiang et al</i>,
<a href="https://arxiv.org/abs/1905.07953"> 📝<i>Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks (KDD'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/ClusterGCN.py"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/ClusterGCN.py"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/ClusterGCN.py"> [🔥PyG Example] </a>
</details>

<!-- 11 -->
<details>
<summary>
<b>DAGNN</b> from <i>Meng Liu et al</i>,
<a href="https://arxiv.org/abs/2007.09296"> 📝<i>Towards Deeper Graph Neural Networks (KDD'20)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/DAGNN.py"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/DAGNN.py"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/DAGNN.py"> [🔥PyG Example] </a>
</details>

### Defense models

<!-- 1 -->
<details>
<summary>
<b>RobustGCN</b> from <i>Petar Veličković et al</i>,
<a href="https://dl.acm.org/doi/10.1145/3292500.3330851"> 📝<i>Robust Graph Convolutional Networks Against Adversarial Attacks (KDD'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/RobustGCN.py"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/RobustGCN.py"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/RobustGCN.py"> [🔥PyG Example] </a>
</details>

<!-- 2 -->
<details>
<summary>
<b>SBVAT</b> from <i>Zhijie Deng et al</i>,
<a href="https://arxiv.org/abs/1902.09192"> 📝<i>Batch Virtual Adversarial Training for Graph Convolutional Networks (ICML'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/SBVAT.py"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/SBVAT.py"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/SBVAT.py"> [🔥PyG Example] </a>
</details>

<!-- 3 -->
<details>
<summary>
<b>OBVAT</b> from <i>Zhijie Deng et al</i>,
<a href="https://arxiv.org/abs/1902.09192"> 📝<i>Batch Virtual Adversarial Training for Graph Convolutional Networks (ICML'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/OBVAT.py"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/OBVAT.py"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/OBVAT.py"> [🔥PyG Example] </a>
</details>

## Unsupervised models

<!-- 1 -->
<details>
<summary>
<b>Deepwalk</b> from <i>Zhijie Deng et al</i>,
<a href="https://arxiv.org/abs/1403.6652"> 📝<i>DeepWalk: Online Learning of Social Representations (KDD'14)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/Deepwalk.py"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/Deepwalk.py"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/Deepwalk.py"> [🔥PyG Example] </a>
</details>

<!-- 2 -->
<details>
<summary>
<b>Node2vec</b> from <i>Zhijie Deng et al</i>,
<a href="https://arxiv.org/abs/1607.00653"> 📝<i>node2vec: Scalable Feature Learning for Networks (KDD'16)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/Node2vec.py"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/Node2vec.py"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/Node2vec.py"> [🔥PyG Example] </a>
</details>

# ⚡ Quick Start
## Datasets
more details please refer to [GraphData](https://github.com/EdisonLeeeee/GraphData).
### Planetoid
fixed datasets
```python
from graphgallery.datasets import Planetoid
# set `verbose=False` to avoid additional outputs 
data = Planetoid('cora', verbose=False)
graph = data.graph
# here `splits` is a dict like instance
splits = data.split_nodes()
# splits.train_nodes:  training indices: 1D Numpy array
# splits.val_nodes:  validation indices: 1D Numpy array
# splits.nodes:  testing indices: 1D Numpy array
>>> graph
Graph(adj_matrix(2708, 2708),
      node_attr(2708, 1433),
      node_label(2708,),
      metadata=None, multiple=False)
```
currently the available datasets are:
```python
>>> data.available_datasets()
('citeseer', 'cora', 'pubmed')
```
### NPZDataset
more scalable datasets (stored with `.npz`)
```python
from graphgallery.datasets import NPZDataset;
# set `verbose=False` to avoid additional outputs
data = NPZDataset('cora', verbose=False, standardize=False)
graph = data.graph
# here `splits` is a dict like instance
splits = data.split_nodes(random_state=42)
>>> graph
Graph(adj_matrix(2708, 2708),
      node_attr(2708, 1433),
      node_label(2708,),
      metadata=None, multiple=False)
```
currently the available datasets are:
```python
>>> data.available_datasets()
('citeseer','citeseer_full','cora','cora_ml','cora_full',
 'amazon_cs','amazon_photo','coauthor_cs','coauthor_phy', 
 'polblogs', 'pubmed', 'flickr','blogcatalog','dblp')
```

## Tensor
+ Strided (dense) Tensor 
```python
>>> backend()
TensorFlow 2.1.2 Backend

>>> from graphgallery import functional as gf
>>> arr = [1, 2, 3]
>>> gf.astensor(arr)
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>

```

+ Sparse Tensor

```python
>>> import scipy.sparse as sp
>>> sp_matrix = sp.eye(3)
>>> gf.astensor(sp_matrix)
<tensorflow.python.framework.sparse_tensor.SparseTensor at 0x7f1bbc205dd8>
```

+ also works for PyTorch, just like

```python
>>> from graphgallery import set_backend
>>> set_backend('torch') # torch, pytorch or th
PyTorch 1.6.0+cu101 Backend

>>> gf.astensor(arr)
tensor([1, 2, 3])

>>> gf.astensor(sp_matrix)
tensor(indices=tensor([[0, 1, 2],
                       [0, 1, 2]]),
       values=tensor([1., 1., 1.]),
       size=(3, 3), nnz=3, layout=torch.sparse_coo)
```

+ To Numpy or Scipy sparse matrix
```python
>>> tensor = gf.astensor(arr)
>>> gf.tensoras(tensor)
array([1, 2, 3])

>>> sp_tensor = gf.astensor(sp_matrix)
>>> gf.tensoras(sp_tensor)
<3x3 sparse matrix of type '<class 'numpy.float32'>'
    with 3 stored elements in Compressed Sparse Row format>
```

+ Or even convert one Tensor to another
```python
>>> tensor = gf.astensor(arr, backend="tensorflow") # or "tf" in short
>>> tensor
<tf.Tensor: shape=(3,), dtype=int64, numpy=array([1, 2, 3])>
>>> gf.tensor2tensor(tensor)
tensor([1, 2, 3])

>>> sp_tensor = gf.astensor(sp_matrix, backend="tensorflow") # set backend="tensorflow" to convert to tensorflow tensor
>>> sp_tensor
<tensorflow.python.framework.sparse_tensor.SparseTensor at 0x7efb6836a898>
>>> gf.tensor2tensor(sp_tensor)
tensor(indices=tensor([[0, 1, 2],
                       [0, 1, 2]]),
       values=tensor([1., 1., 1.]),
       size=(3, 3), nnz=3, layout=torch.sparse_coo)

```

## Example of GCN model
```python
from graphgallery.gallery import GCN

model = GCN(graph, attr_transform="normalize_attr", device="CPU", seed=123)
# build your GCN model with default hyper-parameters
model.build()
# train your model. here splits.train_nodes and splits.val_nodes are numpy arrays
# verbose takes 0, 1, 2, 3, 4
history = model.train(splits.train_nodes, splits.val_nodes, verbose=1, epochs=100)
# test your model
# verbose takes 0, 1, 2
results = model.test(splits.nodes, verbose=1)
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
```
On `Cora` dataset:
```
Training...
100/100 [==============================] - 1s 14ms/step - loss: 1.0161 - acc: 0.9500 - val_loss: 1.4101 - val_acc: 0.7740 - time: 1.4180
Testing...
1/1 [==============================] - 0s 62ms/step - loss: 1.4123 - acc: 0.8120 - time: 0.0620
Test loss 1.4123, Test accuracy 81.20%
```
## Customization
+ Build your model
you can use the following statement to build your model
```python
# one hidden layer with hidden units 32 and activation function RELU
>>> model.build(hiddens=32, activations='relu')

# two hidden layer with hidden units 32, 64 and all activation functions are RELU
>>> model.build(hiddens=[32, 64], activations='relu')

# two hidden layer with hidden units 32, 64 and activation functions RELU and ELU
>>> model.build(hiddens=[32, 64], activations=['relu', 'elu'])

```
+ Train your model
```python
# train with validation
>>> history = model.train(splits.train_nodes, splits.val_nodes, verbose=1, epochs=100)
# train without validation
>>> history = model.train(splits.train_nodes, verbose=1, epochs=100)
```
here `history` is a tensorflow `History` instance.

+ Test you model
```python
>>> results = model.test(splits.nodes, verbose=1)
Testing...
1/1 [==============================] - 0s 62ms/step - loss: 1.4123 - acc: 0.8120 - time: 0.0620
>>> print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
Test loss 1.4123, Test accuracy 81.20%
```

## Visualization
NOTE: you must install [SciencePlots](https://github.com/garrettj403/SciencePlots) package for a better preview.

```python
import matplotlib.pyplot as plt
with plt.style.context(['science', 'no-latex']):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(history.history['accuracy'], label='Training accuracy', linewidth=3)
    axes[0].plot(history.history['val_accuracy'], label='Validation accuracy', linewidth=3)
    axes[0].legend(fontsize=20)
    axes[0].set_title('Accuracy', fontsize=20)
    axes[0].set_xlabel('Epochs', fontsize=20)
    axes[0].set_ylabel('Accuracy', fontsize=20)

    axes[1].plot(history.history['loss'], label='Training loss', linewidth=3)
    axes[1].plot(history.history['val_loss'], label='Validation loss', linewidth=3)
    axes[1].legend(fontsize=20)
    axes[1].set_title('Loss', fontsize=20)
    axes[1].set_xlabel('Epochs', fontsize=20)
    axes[1].set_ylabel('Loss', fontsize=20)
    
    plt.autoscale(tight=True)
    plt.show()        
```
![visualization](https://github.com/EdisonLeeeee/GraphGallery/blob/master/imgs/history.png)

## Using TensorFlow/PyTorch Backend
```python
>>> import graphgallery
>>> graphgallery.backend()
TensorFlow 2.1.0 Backend

>>> graphgallery.set_backend("pytorch")
PyTorch 1.6.0+cu101 Backend
```
###  GCN using PyTorch backend

```python

# The following codes are the same with TensorFlow Backend
>>> from graphgallery.gallery import GCN
>>> model = GCN(graph, attr_transform="normalize_attr", device="GPU", seed=123);
>>> model.build()
>>> history = model.train(splits.train_nodes, splits.val_nodes, verbose=1, epochs=100)
Training...
100/100 [==============================] - 0s 5ms/step - loss: 0.6813 - acc: 0.9214 - val_loss: 1.0506 - val_acc: 0.7820 - time: 0.4734
>>> results = model.test(splits.nodes, verbose=1)
Testing...
1/1 [==============================] - 0s 1ms/step - loss: 1.0131 - acc: 0.8220 - time: 0.0013
>>> print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
Test loss 1.0131, Test accuracy 82.20%

```

# ❓ How to add your datasets
This is motivated by [gnn-benchmark](https://github.com/shchur/gnn-benchmark/)
```python
from graphgallery.data import Graph

# Load the adjacency matrix A, attribute matrix X and labels vector y
# A - scipy.sparse.csr_matrix of shape [n_nodes, n_nodes]
# X - scipy.sparse.csr_matrix or np.ndarray of shape [n_nodes, n_atts]
# y - np.ndarray of shape [n_nodes]

mydataset = Graph(adj_matrix=A, attr_matrix=X, labels=y)
# save dataset
mydataset.to_npz('path/to/mydataset.npz')
# load dataset
mydataset = Graph.from_npz('path/to/mydataset.npz')
```

# ❓ How to define your models

You can follow the codes in the folder `graphgallery.gallery` and write you models based on:

+ TensorFlow
+ PyTorch
+ PyTorch Geometric (PyG)
+ Deep Graph Library (DGL)

NOTE: [PyG](https://github.com/rusty1s/pytorch_geometric) backend and [DGL](https://github.com/dmlc/dgl) backend now are supported in GraphGallery!

```python
>>> import graphgallery
>>> graphgallery.set_backend("pyg")
PyTorch Geometric 1.6.1 (PyTorch 1.6.0+cu101) Backend
```

### GCN using PyG backend

```python
# The following codes are the same with TensorFlow or PyTorch Backend
>>> from graphgallery.gallery import GCN
>>> model = GCN(graph, attr_transform="normalize_attr", device="GPU", seed=123);
>>> model.build()
>>> history = model.train(splits.train_nodes, splits.val_nodes, verbose=1, epochs=100)
Training...
100/100 [==============================] - 0s 3ms/step - loss: 0.5325 - acc: 0.9643 - val_loss: 1.0034 - val_acc: 0.7980 - time: 0.3101
>>> results = model.test(splits.nodes, verbose=1)
Testing...
1/1 [==============================] - 0s 834us/step - loss: 0.9733 - acc: 0.8130 - time: 8.2737e-04
>>> print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
Test loss 0.97332, Test accuracy 81.30%
```



# 😎 More Examples
Please refer to the [examples](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples) directory.

# ⭐ Road Map
- [x] Add PyTorch models support
- [x] Add other frameworks (PyG and DGL) support
- [ ] Add more GNN models (TF and Torch backend)
- [ ] Support for more tasks, e.g., `graph Classification` and `link prediction`
- [x] Support for more types of graphs, e.g., Heterogeneous graph
- [ ] Add Docstrings and Documentation (Building)


# 😘 Acknowledgement
This project is motivated by [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric), [Tensorflow Geometric](https://github.com/CrawlScript/tf_geometric), [Stellargraph](https://github.com/stellargraph/stellargraph) and [DGL](https://github.com/dmlc/dgl), etc., and the original implementations of the authors, thanks for their excellent works!