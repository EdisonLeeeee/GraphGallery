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

+ **ChebyNet** from *Michaël Defferrard et al*, [📝Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), *NIPS'16*, [:octocat:Codes](https://github.com/mdeff/cnn_graph)
+ **GCN** from *Thomas N. Kipf et al*, [📝Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), *ICLR'17*, [:octocat:Codes](https://github.com/tkipf/gcn)
+ **GraphSAGE** from *William L. Hamilton et al*, [📝Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216), *NIPS'17*, [:octocat:Codes](https://github.com/williamleif/GraphSAGE)
+ **FastGCN** from *Jie Chen et al*, [FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling](https://arxiv.org/abs/1801.10247) , *ICLR'18*,[:octocat:Codes](https://github.com/matenure/FastGCN)
+ **LGCN** from  *Hongyang Gao et al*, [📝Large-Scale Learnable Graph Convolutional Networks](https://arxiv.org/abs/1808.03965), *KDD'18*, [:octocat:Codes](https://github.com/divelab/lgcn)
+ **GAT** from *Petar Veličković et al*, [📝Graph Attention Networks](https://arxiv.org/abs/1710.10903), *ICLR'18,* [:octocat:Codes](https://github.com/PetarV-/GAT)
+ **SGC** from *Felix Wu et al*, [📝Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153), *ICML'19*, [:octocat:Codes](https://github.com/Tiiiger/SGC)
+ **GWNN** from *Bingbing Xu et al*, [📝Graph Wavelet Neural Network](https://arxiv.org/abs/1904.07785), *ICLR'19,*[:octocat:Codes](https://github.com/Eilene/GWNN)
+ **GMNN** from *Meng Qu et al*, [📝Graph Markov Neural Networks](https://arxiv.org/abs/1905.06214), *ICML'19,*[:octocat:Codes](https://github.com/DeepGraphLearning/GMNN)
+ **ClusterGCN** from *Wei-Lin Chiang et al*, [📝Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/abs/1905.07953), *KDD'19*, [:octocat:Codes](https://github.com/google-research/google-research/tree/master/cluster_gcn)
+ **DAGNN** from *Meng Liu et al*, [📝Towards Deeper Graph Neural Networks](https://arxiv.org/abs/2007.09296), *KDD'20*, [:octocat:Codes](https://github.com/mengliu1998/DeeperGNN)

## Defense models
+ **RobustGCN** from *Dingyuan Zhu et al*, [📝Robust Graph Convolutional Networks Against Adversarial Attacks](https://dl.acm.org/doi/10.1145/3292500.3330851), *KDD'19*, [:octocat:Codes](https://github.com/thumanlab/nrlweb/blob/master/static/assets/download/RGCN.zip)
+ **SBVAT/OBVAT** from *Zhijie Deng et al*, [📝Batch Virtual Adversarial Training for Graph Convolutional Networks](https://arxiv.org/abs/1902.09192), *ICML'19*, [:octocat:Codes](https://github.com/thudzj/BVAT)


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

# Acknowledgement
This project is motivated by [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric), [Tensorflow Geometric](https://github.com/CrawlScript/tf_geometric) and [Stellargraph](https://github.com/stellargraph/stellargraph), and the original implementations from the authors, thanks for their excellent works!

