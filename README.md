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
  <img alt="stars" src="https://img.shields.io/github/stars/EdisonLeeeee/GraphGallery">
  <img alt="forks" src="https://img.shields.io/github/forks/EdisonLeeeee/GraphGallery">
  <img alt="issues" src="https://img.shields.io/github/issues/EdisonLeeeee/GraphGallery">    
  <a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/EdisonLeeeee/GraphGallery" alt="pypi">
  </a>       
</p>



# GraphGallery
GraphGallery is a gallery for benchmark graph neural networks with [TensorFlow 2.x](https://github.com/tensorflow/tensorflow) and [PyTorch](https://github.com/pytorch/pytorch) backend. GraphGallery 0.5.x is a total re-write from previous versions, and some things have changed. 

# 👀 What's important
Difference between GraphGallery and [Pytorch Geometric (PyG)](https://github.com/rusty1s/pytorch_geometric), [Deep Graph Library (DGL)](https://github.com/dmlc/dgl), etc...
+ PyG and DGL are just like **TensorFlow** while GraphGallery is more like **Keras**
+ GraphGallery is more extensible and user-friendly
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
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_ChebyNet.ipynb"> :octocat:TensorFLow Example</a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_ChebyNet.ipynb"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/test_ChebyNet.ipynb"> [🔥PyG Example] </a>

</details>

<!-- 2 -->

<details>
<summary>
<b>GCN</b> from <i>Thomas N. Kipf et al</i>,
<a href="https://arxiv.org/abs/1609.02907"> 📝<i>Semi-Supervised Classification with Graph Convolutional Networks (ICLR'17)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_GCN.ipynb"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_GCN.ipynb"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/test_GCN.ipynb"> [🔥PyG Example] </a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/DGL-PyTorch/test_GCN.ipynb"> [🔥DGL-PyTorch Example] </a>
</details>

<!-- 3 -->
<details>
<summary>
<b>GraphSAGE</b> from <i>William L. Hamilton et al</i>,
<a href="https://arxiv.org/abs/1706.02216"> 📝<i>Inductive Representation Learning on Large Graphs (NeurIPS'17)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_GraphSAGE.ipynb"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_GraphSAGE.ipynb"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/test_GraphSAGE.ipynb"> [🔥PyG Example] </a>
</details>

<!-- 4 -->
<details>
<summary>
<b>FastGCN</b> from <i>Jie Chen et al</i>,
<a href="https://arxiv.org/abs/1801.10247"> 📝<i>FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling (ICLR'18)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_FastGCN.ipynb"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_FastGCN.ipynb"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/test_FastGCN.ipynb"> [🔥PyG Example] </a>
</details>

<!-- 5 -->
<details>
<summary>
<b>LGCN</b> from <i>Hongyang Gao et al</i>,
<a href="https://arxiv.org/abs/1808.03965"> 📝<i>Large-Scale Learnable Graph Convolutional Networks (KDD'18)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_LGCN.ipynb"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_LGCN.ipynb"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/test_LGCN.ipynb"> [🔥PyG Example] </a>
</details>

<!-- 6 -->
<details>
<summary>
<b>GAT</b> from <i>Petar Veličković et al</i>,
<a href="https://arxiv.org/abs/1710.10903"> 📝<i>Graph Attention Networks (ICLR'18)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_GAT.ipynb"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_GAT.ipynb"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/test_GAT.ipynb"> [🔥PyG Example] </a>
</details>

<!-- 7 -->
<details>
<summary>
<b>SGC</b> from <i>Felix Wu et al</i>,
<a href="https://arxiv.org/abs/1902.07153"> 📝<i>Simplifying Graph Convolutional Networks (ICLR'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_SGC.ipynb"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_SGC.ipynb"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/test_SGC.ipynb"> [🔥PyG Example] </a>
</details>

<!-- 8 -->
<details>
<summary>
<b>GWNN</b> from <i>Bingbing Xu et al</i>,
<a href="https://arxiv.org/abs/1904.07785"> 📝<i>Graph Wavelet Neural Network (ICLR'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_GWNN.ipynb"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_GWNN.ipynb"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/test_GWNN.ipynb"> [🔥PyG Example] </a>
</details>

<!-- 69 -->
<details>
<summary>
<b>GMNN</b> from <i>Meng Qu et al</i>,
<a href="https://arxiv.org/abs/1905.06214"> 📝<i>Graph Attention Networks (ICLR'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_GMNN.ipynb"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_GMNN.ipynb"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/test_GMNN.ipynb"> [🔥PyG Example] </a>
</details>

<!-- 10 -->
<details>
<summary>
<b>ClusterGCN</b> from <i>Wei-Lin Chiang et al</i>,
<a href="https://arxiv.org/abs/1905.07953"> 📝<i>Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks (KDD'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_ClusterGCN.ipynb"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_ClusterGCN.ipynb"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/test_ClusterGCN.ipynb"> [🔥PyG Example] </a>
</details>

<!-- 11 -->
<details>
<summary>
<b>DAGNN</b> from <i>Meng Liu et al</i>,
<a href="https://arxiv.org/abs/2007.09296"> 📝<i>Towards Deeper Graph Neural Networks (KDD'20)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_DAGNN.ipynb"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_DAGNN.ipynb"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/test_DAGNN.ipynb"> [🔥PyG Example] </a>
</details>

### Defense models

<!-- 1 -->
<details>
<summary>
<b>RobustGCN</b> from <i>Petar Veličković et al</i>,
<a href="https://dl.acm.org/doi/10.1145/3292500.3330851"> 📝<i>Robust Graph Convolutional Networks Against Adversarial Attacks (KDD'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_RobustGCN.ipynb"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_RobustGCN.ipynb"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/test_RobustGCN.ipynb"> [🔥PyG Example] </a>
</details>

<!-- 2 -->
<details>
<summary>
<b>SBVAT</b> from <i>Zhijie Deng et al</i>,
<a href="https://arxiv.org/abs/1902.09192"> 📝<i>Batch Virtual Adversarial Training for Graph Convolutional Networks (ICML'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_SBVAT.ipynb"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_SBVAT.ipynb"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/test_SBVAT.ipynb"> [🔥PyG Example] </a>
</details>

<!-- 3 -->
<details>
<summary>
<b>OBVAT</b> from <i>Zhijie Deng et al</i>,
<a href="https://arxiv.org/abs/1902.09192"> 📝<i>Batch Virtual Adversarial Training for Graph Convolutional Networks (ICML'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_OBVAT.ipynb"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_OBVAT.ipynb"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/test_OBVAT.ipynb"> [🔥PyG Example] </a>
</details>

## Unsupervised models

<!-- 1 -->
<details>
<summary>
<b>Deepwalk</b> from <i>Zhijie Deng et al</i>,
<a href="https://arxiv.org/abs/1403.6652"> 📝<i>DeepWalk: Online Learning of Social Representations (KDD'14)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_Deepwalk.ipynb"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_Deepwalk.ipynb"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/test_Deepwalk.ipynb"> [🔥PyG Example] </a>
</details>

<!-- 2 -->
<details>
<summary>
<b>Node2vec</b> from <i>Zhijie Deng et al</i>,
<a href="https://arxiv.org/abs/1607.00653"> 📝<i>node2vec: Scalable Feature Learning for Networks (KDD'16)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_Node2vec.ipynb"> [:octocat:TensorFLow Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_Node2vec.ipynb"> [🔥PyTorch Example]</a>,
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyG/test_Node2vec.ipynb"> [🔥PyG Example] </a>
</details>

# ⚡ Quick Start
## Datasets
more details please refer to [GraphData](https://github.com/EdisonLeeeee/GraphData).
### Planetoid
fixed datasets
```python
from graphgallery.data import Planetoid
# set `verbose=False` to avoid additional outputs 
data = Planetoid('cora', verbose=False)
graph = data.graph
idx_train, idx_val, idx_test = data.split()
# idx_train:  training indices: 1D Numpy array
# idx_val:  validation indices: 1D Numpy array
# idx_test:  testing indices: 1D Numpy array
>>> graph
Graph(adj_matrix(2708, 2708), attr_matrix(2708, 2708), labels(2708,))
```
currently the supported datasets are:
```python
>>> data.supported_datasets
('citeseer', 'cora', 'pubmed')
```
### NPZDataset
more scalable datasets (stored with `.npz`)
```python
from graphgallery.data import NPZDataset;
# set `verbose=False` to avoid additional outputs
data = NPZDataset('cora', verbose=False, standardize=False)
graph = data.graph
idx_train, idx_val, idx_test = data.split(random_state=42)
>>> graph
Graph(adj_matrix(2708, 2708), attr_matrix(2708, 2708), labels(2708,))
```
currently the supported datasets are:
```python
>>> data.supported_datasets
('citeseer','citeseer_full','cora','cora_ml','cora_full',
 'amazon_cs','amazon_photo','coauthor_cs','coauthor_phy', 
 'polblogs', 'pubmed', 'flickr','blogcatalog','dblp')
```

## Tensor
+ Strided (dense) Tensor 
```python
>>> backend()
TensorFlow 2.1.2 Backend

>>> from graphgallery import functional as F
>>> arr = [1, 2, 3]
>>> F.astensor(arr)
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>

```

+ Sparse Tensor

```python
>>> import scipy.sparse as sp
>>> sp_matrix = sp.eye(3)
>>> F.astensor(sp_matrix)
<tensorflow.python.framework.sparse_tensor.SparseTensor at 0x7f1bbc205dd8>
```

+ also works for PyTorch, just like

```python
>>> from graphgallery import set_backend
>>> set_backend('torch') # torch, pytorch or th
PyTorch 1.6.0+cu101 Backend

>>> F.astensor(arr)
tensor([1, 2, 3])

>>> F.astensor(sp_matrix)
tensor(indices=tensor([[0, 1, 2],
                       [0, 1, 2]]),
       values=tensor([1., 1., 1.]),
       size=(3, 3), nnz=3, layout=torch.sparse_coo)
```

+ To Numpy or Scipy sparse matrix
```python
>>> tensor = F.astensor(arr)
>>> F.tensoras(tensor)
array([1, 2, 3])

>>> sp_tensor = F.astensor(sp_matrix)
>>> F.tensoras(sp_tensor)
<3x3 sparse matrix of type '<class 'numpy.float32'>'
    with 3 stored elements in Compressed Sparse Row format>
```

+ Or even convert one Tensor to another one
```python
>>> tensor = F.astensor(arr, backend="tensorflow") # or "tf" in short
>>> tensor
<tf.Tensor: shape=(3,), dtype=int64, numpy=array([1, 2, 3])>
>>> F.tensor2tensor(tensor)
tensor([1, 2, 3])

>>> sp_tensor = F.astensor(sp_matrix, backend="tensorflow") # set backend="tensorflow" to convert to tensorflow tensor
>>> sp_tensor
<tensorflow.python.framework.sparse_tensor.SparseTensor at 0x7efb6836a898>
>>> F.tensor2tensor(sp_tensor)
tensor(indices=tensor([[0, 1, 2],
                       [0, 1, 2]]),
       values=tensor([1., 1., 1.]),
       size=(3, 3), nnz=3, layout=torch.sparse_coo)

```

## Example of GCN model
```python
from graphgallery.nn.gallery import GCN

model = GCN(graph, attr_transform="normalize_attr", device="CPU", seed=123)
# build your GCN model with default hyper-parameters
model.build()
# train your model. here idx_train and idx_val are numpy arrays
# verbose takes 0, 1, 2, 3, 4
his = model.train(idx_train, idx_val, verbose=1, epochs=100)
# test your model
# verbose takes 0, 1, 2
loss, accuracy = model.test(idx_test, verbose=1)
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```
On `Cora` dataset:
```
Training...
100/100 [==============================] - 1s 14ms/step - loss: 1.0161 - acc: 0.9500 - val_loss: 1.4101 - val_acc: 0.7740 - time: 1.4180
Testing...
1/1 [==============================] - 0s 62ms/step - test_loss: 1.4123 - test_acc: 0.8120 - time: 0.0620
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
>>> his = model.train(idx_train, idx_val, verbose=1, epochs=100)
# train without validation
>>> his = model.train(idx_train, verbose=1, epochs=100)
```
here `his` is a tensorflow `History` instance.

+ Test you model
```python
>>> loss, accuracy = model.test(idx_test, verbose=1)
Testing...
1/1 [==============================] - 0s 62ms/step - test_loss: 1.4123 - test_acc: 0.8120 - time: 0.0620
>>> print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
Test loss 1.4123, Test accuracy 81.20%
```

## Visualization
NOTE: you must install [SciencePlots](https://github.com/garrettj403/SciencePlots) package for a better preview.

```python
import matplotlib.pyplot as plt
with plt.style.context(['science', 'no-latex']):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(his.history['acc'], label='Train accuracy', linewidth=3)
    axes[0].plot(his.history['val_acc'], label='Val accuracy', linewidth=3)
    axes[0].legend(fontsize=20)
    axes[0].set_title('Accuracy', fontsize=20)
    axes[0].set_xlabel('Epochs', fontsize=20)
    axes[0].set_ylabel('Accuracy', fontsize=20)

    axes[1].plot(his.history['loss'], label='Training loss', linewidth=3)
    axes[1].plot(his.history['val_loss'], label='Validation loss', linewidth=3)
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
>>> from graphgallery.nn.gallery import GCN
>>> model = GCN(graph, attr_transform="normalize_attr", device="GPU", seed=123);
>>> model.build()
>>> his = model.train(idx_train, idx_val, verbose=1, epochs=100)
Training...
100/100 [==============================] - 0s 5ms/step - loss: 0.6813 - acc: 0.9214 - val_loss: 1.0506 - val_acc: 0.7820 - time: 0.4734
>>> loss, accuracy = model.test(idx_test, verbose=1)
Testing...
1/1 [==============================] - 0s 1ms/step - test_loss: 1.0131 - test_acc: 0.8220 - time: 0.0013
>>> print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
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

You can follow the codes in the folder `graphgallery.nn.gallery` and write you models based on:

+ TensorFlow
+ PyTorch
+ PyTorch Geometric (PyG)
+ Deep Graph Library (DGL)

### NOTE: [PyG](https://github.com/rusty1s/pytorch_geometric) backend and [DGL](https://github.com/dmlc/dgl) backend now are supported in GraphGallery!

```python
>>> import graphgallery
>>> graphgallery.set_backend("pyg")
PyTorch Geometric 1.6.1 (PyTorch 1.6.0+cu101) Backend
```

### GCN using PyG backend

```python
# The following codes are the same with TensorFlow or PyTorch Backend
>>> from graphgallery.nn.gallery import GCN
>>> model = GCN(graph, attr_transform="normalize_attr", device="GPU", seed=123);
>>> model.build()
>>> his = model.train(idx_train, idx_val, verbose=1, epochs=100)
Training...
100/100 [==============================] - 0s 3ms/step - loss: 0.5325 - acc: 0.9643 - val_loss: 1.0034 - val_acc: 0.7980 - time: 0.3101
>>> loss, accuracy = model.test(idx_test, verbose=1)
Testing...
1/1 [==============================] - 0s 834us/step - test_loss: 0.9733 - test_acc: 0.8130 - time: 8.2737e-04
>>> print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
Test loss 0.97332, Test accuracy 81.30%
```



# 😎 More Examples
Please refer to the [examples](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples) directory.

# ⭐ TODO List
- [x] Add PyTorch models support
- [x] Add other frameworks (PyG and DGL) support
- [ ] Add more GNN models (TF and Torch backend)
- [ ] Support for more tasks, e.g., `graph Classification` and `link prediction`
- [ ] Support for more types of graphs, e.g., Heterogeneous graph
- [ ] Add Docstrings and Documentation (Building)


# 😘 Acknowledgement
This project is motivated by [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric), [Tensorflow Geometric](https://github.com/CrawlScript/tf_geometric) and [Stellargraph](https://github.com/stellargraph/stellargraph), [DGL](https://github.com/dmlc/dgl), etc., and the original implementations of the authors, thanks for their excellent works!