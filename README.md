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
    <img src="https://img.shields.io/badge/PyTorch->=1.5-FF6F00?logo=pytorch" alt="pytorch">
  </a>   
  <a href="https://pypi.org/project/graphgallery/">
    <img src="https://badge.fury.io/py/graphgallery.svg" alt="pypi">
  </a>       
  <a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/EdisonLeeeee/GraphGallery" alt="license">
  </a>       
</p>

- [GraphGallery](#graphgallery)
- [👀 What's important](#-whats-important)
- [🚀 Installation](#-installation)
- [🤖 Implementations](#-implementations)
- [⚡ Quick Start](#-quick-start)
  - [Datasets](#datasets)
    - [Planetoid](#planetoid)
    - [NPZDataset](#npzdataset)
  - [Framework-neutral Tensor](#framework-neutra-tensor)
  - [Example of GCN model](#example-of-gcn-model)
  - [Customization](#customization)
  - [Visualization](#visualization)
  - [Using TensorFlow/PyTorch Backend](#using-tensorflowpytorch-backend)
    - [GCN using PyTorch backend](#gcn-using-pytorch-backend)
- [❓ How to add your datasets](#-how-to-add-your-datasets)
- [❓ How to define your models](#-how-to-define-your-models)
    - [GCN using PyG backend](#gcn-using-pyg-backend)
- [⭐ Road Map](#-road-map)
- [😘 Acknowledgement](#-acknowledgement)

# GraphGallery
GraphGallery is a gallery for benchmarking Graph Neural Networks (GNNs) with [TensorFlow 2.x](https://github.com/tensorflow/tensorflow) and [PyTorch](https://github.com/pytorch/pytorch) backend. GraphGallery 0.6.x is a total re-write from previous versions, and some things have changed. 

NEWS: 
+ [PyG](https://github.com/rusty1s/pytorch_geometric) backend and [DGL](https://github.com/dmlc/dgl) backend now are available in GraphGallery
+ GraphGallery now supports Multiple Graph for different tasks


# 👀 What's important
Differences between GraphGallery and [Pytorch Geometric (PyG)](https://github.com/rusty1s/pytorch_geometric), [Deep Graph Library (DGL)](https://github.com/dmlc/dgl), etc...
+ PyG and DGL are just like **TensorFlow** while GraphGallery is more like **Keras**
+ GraphGallery is a plug-and-play and user-friendly toolbox
+ GraphGallery has high scalaibility for researchers and developers to use


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

GraphGallery has been tested on:
+ CUDA 10.1
+ TensorFlow 2.1~2.3, 2.4 is unavailable now and 2.1.2 is recommended.
+ PyTorch 1.5~1.7
+ Pytorch Geometric (PyG) 1.6.1
+ DGL 0.5.2, 0.5.3

# 🤖 Implementations
Please refer to the [examples](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples) directory.

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
# splits.test_nodes:  testing indices: 1D Numpy array
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
data = NPZDataset('cora', verbose=False)
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

## Framework-neutral Tensor
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
100/100 [==============================] - 1s 14ms/step - loss: 1.0161 - accuracy: 0.9500 - val_loss: 1.4101 - val_accuracy: 0.7740 - Dur.: 1.4180
Testing...
1/1 [==============================] - 0s 62ms/step - loss: 1.4123 - accuracy: 0.8120 - Dur.: 0.0620
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
>>> results = model.test(splits.test_nodes, verbose=1)
Testing...
1/1 [==============================] - 0s 62ms/step - loss: 1.4123 - accuracy: 0.8120 - Dur.: 0.0620
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
    axes[0].plot(history.history['val_accuracyuracy'], label='Validation accuracy', linewidth=3)
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
TensorFlow 2.1.2 Backend

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
100/100 [==============================] - 0s 5ms/step - loss: 0.6813 - accuracy: 0.9214 - val_loss: 1.0506 - val_accuracy: 0.7820 - Dur.: 0.4734
>>> results = model.test(splits.test_nodes, verbose=1)
Testing...
1/1 [==============================] - 0s 1ms/step - loss: 1.0131 - accuracy: 0.8220 - Dur.: 0.0013
>>> print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
Test loss 1.0131, Test accuracy 82.20%

```

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
100/100 [==============================] - 0s 3ms/step - loss: 0.5325 - accuracy: 0.9643 - val_loss: 1.0034 - val_accuracy: 0.7980 - Dur.: 0.3101
>>> results = model.test(splits.test_nodes, verbose=1)
Testing...
1/1 [==============================] - 0s 834us/step - loss: 0.9733 - accuracy: 0.8130 - Dur.: 8.2737e-04
>>> print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
Test loss 0.97332, Test accuracy 81.30%
```
similarly, you can use DGL backend just by:
```python
# DGL PyTorch backend
>>> graphgallery.set_backend("dgl")
# DGL TensorFlow backend
>>> graphgallery.set_backend("dgl-tf")

```



# ⭐ Road Map
- [x] Add PyTorch models support
- [x] Add other frameworks (PyG and DGL) support
- [ ] Add more GNN models (TF and Torch backend)
- [ ] Support for more tasks, e.g., `graph Classification` and `link prediction`
- [x] Support for more types of graphs, e.g., Heterogeneous graph
- [ ] Add Docstrings and Documentation (Building)
- [ ] Comprehensive tutorials

# 😘 Acknowledgement
This project is motivated by [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric), [Tensorflow Geometric](https://github.com/CrawlScript/tf_geometric), [Stellargraph](https://github.com/stellargraph/stellargraph) and [DGL](https://github.com/dmlc/dgl), etc., and the original implementations of the authors, thanks for their excellent works!