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
GraphGallery is a gallery of state-of-the-arts graph neural networks for [TensorFlow 2.x](https://github.com/tensorflow/tensorflow) and [PyTorch](https://github.com/pytorch/pytorch). GraphGallery 0.3.0 is a total re-write from 0.2.0, and some things have changed. 
<!-- 
This repo aims to achieve 4 goals:
+ Similar or higher performance
+ Faster training and testing
+ Simple and convenient to use, high scalability
+ Easy to read source codes -->

# Installation
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
# Implementations
In detail, the following methods are currently implemented:

## Semi-supervised models
### General models

+ **ChebyNet** from *Michaël Defferrard et al*, [📝Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), *NIPS'16*. 
 [[🌋 TF]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_ChebyNet.ipynb)
+ **GCN** from *Thomas N. Kipf et al*, [📝Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), *ICLR'17*. 
 [[🌋 TF]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_GCN.ipynb), [[🔥 Torch]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_GCN.ipynb)
+ **GraphSAGE** from *William L. Hamilton et al*, [📝Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216), *NIPS'17*. 
 [[🌋 TF]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_GraphSAGE.ipynb)
+ **FastGCN** from *Jie Chen et al*, [FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling](https://arxiv.org/abs/1801.10247), *ICLR'18*. 
 [[🌋 TF]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_FastGCN.ipynb)
+ **LGCN** from  *Hongyang Gao et al*, [📝Large-Scale Learnable Graph Convolutional Networks](https://arxiv.org/abs/1808.03965), *KDD'18*. 
 [[🌋 TF]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_LGCN.ipynb)
+ **GAT** from *Petar Veličković et al*, [📝Graph Attention Networks](https://arxiv.org/abs/1710.10903), *ICLR'18*. 
 ), [[🌋 TF]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_GAT.ipynb), [[🔥 Torch]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_GAT.ipynb)
+ **SGC** from *Felix Wu et al*, [📝Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153), *ICML'19*. 
 [[🌋 TF]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_SGC.ipynb),  [[🔥 Torch]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_SGC.ipynb)
+ **GWNN** from *Bingbing Xu et al*, [📝Graph Wavelet Neural Network](https://arxiv.org/abs/1904.07785), *ICLR'19*. 
 [[🌋 TF]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_GWNN.ipynb)
+ **GMNN** from *Meng Qu et al*, [📝Graph Markov Neural Networks](https://arxiv.org/abs/1905.06214), *ICML'19*. 
 [[🌋 TF]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_GMNN.ipynb)
+ **ClusterGCN** from *Wei-Lin Chiang et al*, [📝Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/abs/1905.07953), *KDD'19*. 
[[🌋 TF]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_ClusterGCN.ipynb), [[🔥 Torch]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_ClusterGCN.ipynb)
+ **DAGNN** from *Meng Liu et al*, [📝Towards Deeper Graph Neural Networks](https://arxiv.org/abs/2007.09296), *KDD'20*. 
 [[🌋 TF]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_DAGNN.ipynb)


### Defense models
+ **RobustGCN** from *Dingyuan Zhu et al*, [📝Robust Graph Convolutional Networks Against Adversarial Attacks](https://dl.acm.org/doi/10.1145/3292500.3330851), *KDD'19*. 
  [[🌋 TF]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_RobustGCN.ipynb)
+ **SBVAT/OBVAT** from *Zhijie Deng et al*, [📝Batch Virtual Adversarial Training for Graph Convolutional Networks](https://arxiv.org/abs/1902.09192), *ICML'19*. 
 [[🌋 TF]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_SBVAT.ipynb), [[🌋 TF]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_OBVAT.ipynb)

## Unsupervised models
+ **Deepwalk** from *Bryan Perozzi et al*, [📝DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652), *KDD'14*. 
 [[🌋 TF]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_Deepwalk.ipynb)
+ **Node2vec** from *Aditya Grover et al*, [📝node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653), *KDD'16*. 
[[🌋 TF]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_Node2vec.ipynb)

# Quick Start
## Datasets
```python
from graphgallery.data import Planetoid
# set `verbose=False` to avoid these printed tables
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

## Example of GCN model
```python
from graphgallery.nn.models import GCN

model = GCN(graph, attr_transform="normalize_attr", device="CPU", seed=123)
# build your GCN model with default hyper-parameters
model.build()
# train your model. here idx_train and idx_val are numpy arrays
his = model.train(idx_train, idx_val, verbose=1, epochs=100)
# test your model
loss, accuracy = model.test(idx_test)
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```
On `Cora` dataset:
```
<Loss = 1.0161 Acc = 0.9500 Val_Loss = 1.4101 Val_Acc = 0.7740 >: 100%|██████████| 100/100 [00:01<00:00, 118.02it/s]
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
here `his` is tensorflow `History` (like) instance.

+ Test you model
```python
>>> loss, accuracy = model.test(idx_test)
>>> print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
Test loss 1.4124, Test accuracy 81.20%
```

## Visualization
NOTE: you must install [SciencePlots](https://github.com/garrettj403/SciencePlots) package for a better preview.

```python
import matplotlib.pyplot as plt
with plt.style.context(['science', 'no-latex']):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(his.history['acc'], label='Train accuracy')
    axes[0].plot(his.history['val_acc'], label='Val accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].legend()

    axes[1].plot(his.history['loss'], label='Training loss')
    axes[1].plot(his.history['val_loss'], label='Validation loss')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()
    
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
GCN using PyTorch backend
```python

# The following codes are the same with TensorFlow Backend
>>> from graphgallery.nn.models import GCN
>>> model = GCN(graph, attr_transform="normalize_attr", device="GPU", seed=123);
>>> model.build()
>>> his = model.train(idx_train, idx_val, verbose=1, epochs=100)
loss 0.57, acc 96.43%, val_loss 1.04, val_acc 78.20%: 100%|██████████| 100/100 [00:00<00:00, 210.90it/s]
>>> loss, accuracy = model.test(idx_test)
>>> print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
Test loss 1.0271, Test accuracy 81.10%

```

# How to add your custom datasets
TODO

# How to define your custom models
TODO

# More Examples
Please refer to the [examples](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples) directory.

# TODO Lists
- [ ] Add Docstrings and Documentation (Building)
- [x] Add PyTorch models support
- [ ] Support for `graph Classification` and `link prediction` tasks
- [ ] Support for Heterogeneous graphs


# Acknowledgement
This project is motivated by [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric), [Tensorflow Geometric](https://github.com/CrawlScript/tf_geometric) and [Stellargraph](https://github.com/stellargraph/stellargraph), and the original implementations of the authors, thanks for their excellent works!

