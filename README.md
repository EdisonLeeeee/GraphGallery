# GraphGallery

[pypi-image]: https://badge.fury.io/py/graphgallery.svg
[pypi-url]: https://pypi.org/project/graphgallery/

<p align="center">
  <img width = "500" height = "300" src="https://github.com/EdisonLeeeee/GraphGallery/blob/master/imgs/graphgallery.svg" alt="logo"/>
</p>

---
![Python](https://img.shields.io/badge/python-%3E%3D3.6-blue)
![tensorflow](https://img.shields.io/badge/tensorflow-%3E%3D2.1.0-orange)
[![PyPI Version][pypi-image]][pypi-url]
![](https://img.shields.io/github/stars/EdisonLeeeee/GraphGallery)
![](https://img.shields.io/github/forks/EdisonLeeeee/GraphGallery)
![](https://img.shields.io/github/issues/EdisonLeeeee/GraphGallery)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A gallery of state-of-the-arts graph neural networks. Implemented with [TensorFlow](https://github.com/tensorflow/tensorflow) 2.x.


This repo aims to achieve 4 goals:
+ Similar or higher performance
+ Faster training and testing
+ Simple and convenient to use, high scalability
+ Easy to read source codes

# Requirements

+ python>=3.6
+ tensorflow>=2.1 (2.1 is recommended)
+ networkx==2.3
+ scipy
+ scikit_learn
+ numpy
+ numba
+ gensim

Other packages (not necessary):

+ metis==0.2a4 (required for `ClusterGCN`)
+ texttable

# Installation
```bash
pip install -U graphgallery
```

# Implementation
## General models
+ **GCN** from *Semi-Supervised Classification with Graph Convolutional Networks*, *ICLR'17*, [📝Paper](https://arxiv.org/abs/1609.02907) [:octocat:Official Implementation](https://github.com/tkipf/gcn)
+ **GAT** from *Graph Attention Networks*, *ICLR'18,* [📝Paper](https://arxiv.org/abs/1710.10903>) [:octocat:Official Implementation](https://github.com/PetarV-/GAT)
+ **SGC** from *Simplifying Graph Convolutional Networks*, *ICML'19*, [📝Paper](https://arxiv.org/abs/1902.07153) [:octocat:Official Implementation](https://github.com/Tiiiger/SGC)
+ **GraphSAGE** from *Inductive Representation Learning on Large Graphs*, *NIPS'17*, [📝Paper](https://arxiv.org/abs/1706.02216) [:octocat:Official Implementation](https://github.com/williamleif/GraphSAGE)
+ **GWNN** from *Graph Wavelet Neural Network*, *ICLR'19,* [📝Paper](https://arxiv.org/abs/1904.07785) [:octocat:Official Implementation](https://github.com/Eilene/GWNN)
+ **GMNN** from *Graph Markov Neural Networks*, *ICML'19,* [📝Paper](https://arxiv.org/abs/1905.06214>) [:octocat:Official Implementation](https://github.com/DeepGraphLearning/GMNN)
+ **ChebyNet** from *Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering*, *NIPS'16*, [📝Paper](https://arxiv.org/abs/1606.09375>) [:octocat:Official Implementation](https://github.com/mdeff/cnn_graph)
+ **ClusterGCN** from *Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks*, *KDD'19*, [📝Paper](https://arxiv.org/abs/1905.07953) [:octocat:Official Implementation](https://github.com/google-research/google-research/tree/master/cluster_gcn)
+ **FastGCN** from *FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling*, *ICLR'18*, [📝Paper](https://arxiv.org/abs/1801.10247) [:octocat:Official Implementation](https://github.com/matenure/FastGCN)
+ **LGCN** from  *Large-Scale Learnable Graph Convolutional Networks*, *KDD'18*, [📝Paper](https://arxiv.org/abs/1808.03965) [:octocat:Official Implementation](https://github.com/divelab/lgcn)
+ **DAGNN** from *Towards Deeper Graph Neural Networks*, *KDD'20*, [📝Paper](https://arxiv.org/abs/2007.09296>) [:octocat:Official Implementation](https://github.com/mengliu1998/DeeperGNN)

## Defense models
+ **RobustGCN** from *Robust Graph Convolutional Networks Against Adversarial Attacks*, [📝Paper](https://dl.acm.org/doi/10.1145/3292500.3330851) [:octocat:Official Implementation](https://github.com/thumanlab/nrlweb/blob/master/static/assets/download/RGCN.zip)
+ **SBVAT/OBVAT** from *Batch Virtual Adversarial Training for Graph Convolutional Networks*, [📝Paper](https://arxiv.org/abs/1902.09192) [:octocat:Official Implementation](https://github.com/thudzj/BVAT)


# Quick Start
## Example of GCN model
```python
from graphgallery.nn.models import GCN
# adj is scipy sparse matrix, x is numpy array matrix
model = GCN(adj, x, labels, device='GPU', norm_x='l1', seed=123)
# build your GCN model with custom hyper-parameters
model.build()
# train your model. here idx_train and idx_val are numpy arrays
his = model.train(idx_train, idx_val, verbose=1, epochs=100)
# test your model
loss, accuracy = model.test(idx_test)
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```
On `Cora` dataset:
```
loss 1.02, acc 95.00%, val_loss 1.41, val_acc 77.40%: 100%|██████████| 100/100 [00:02<00:00, 37.07it/s]
Test loss 1.4123, Test accuracy 81.20%
```

## Build your model
you can use the following statement to build your model
```python
# one hidden layer with hidden units 32 and activation function RELU
>>> model.build(hiddens=32, activations='relu')

# two hidden layer with hidden units 32, 64 and all activation functions are RELU
>>> model.build(hiddens=[32, 64], activations='relu')

# two hidden layer with hidden units 32, 64 and activation functions RELU and ELU
>>> model.build(hiddens=[32, 64], activations=['relu', 'elu'])

# other parameters like `dropouts` and `l2_norms` (if have) are the SAME.
```
## Train or test your model
More details can be seen in the methods [model.train](https://github.com/EdisonLeeeee/GraphGallery/blob/master/graphgallery/nn/models/semisupervised/semi_supervised_model.py#L80) and [model.test](https://github.com/EdisonLeeeee/GraphGallery/blob/master/graphgallery/nn/models/semisupervised/semi_supervised_model.py#L382) 

## Hyper-parameters
you can simply use `model.show()` to show all your `Hyper-parameters`.
Otherwise you can also use `model.show('model')` or `model.show('train')` to show your model parameters and training parameters.
NOTE: you should install texttable first.

## Visualization
+ Accuracy
```python
import matplotlib.pyplot as plt
plt.plot(his.history['acc'])
plt.plot(his.history['val_acc'])
plt.legend(['Accuracy', 'Val Accuracy'])
plt.xlabel('Epochs')
plt.show()
```
![visualization](https://github.com/EdisonLeeeee/GraphGallery/blob/master/imgs/visualization_acc.png)

+ Loss
```python
import matplotlib.pyplot as plt
plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.legend(['Loss', 'Val Loss'])
plt.xlabel('Epochs')
plt.show()
```
![visualization](https://github.com/EdisonLeeeee/GraphGallery/blob/master/imgs/visualization_loss.png)


