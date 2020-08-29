# GraphGallery

[pypi-image]: https://badge.fury.io/py/graphgallery.svg
[pypi-url]: https://pypi.org/project/graphgallery/

<p align="center">
  <img width = "500" height = "300" src="https://github.com/EdisonLeeeee/GraphGallery/blob/master/imgs/graphgallery.svg" alt="logo"/>
</p>

---
[![Python 3.6](https://img.shields.io/badge/Python->=3.6-3776AB)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow >=2.1](https://img.shields.io/badge/TensorFlow->=2.1-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.1.0)
[![PyPI Version][pypi-image]][pypi-url]
![](https://img.shields.io/github/stars/EdisonLeeeee/GraphGallery)
![](https://img.shields.io/github/forks/EdisonLeeeee/GraphGallery)
![](https://img.shields.io/github/issues/EdisonLeeeee/GraphGallery)
[![GitHub license](https://img.shields.io/github/license/EdisonLeeeee/GraphGallery)](https://github.com/EdisonLeeeee/GraphGallery/blob/master/LICENSE)

GraphGallery is a gallery of state-of-the-arts graph neural networks for [TensorFlow](https://github.com/tensorflow/tensorflow) 2.x.


This repo aims to achieve 4 goals:
+ Similar or higher performance
+ Faster training and testing
+ Simple and convenient to use, high scalability
+ Easy to read source codes
---

# Installation
+ Build from source
```bash
git clone https://github.com/EdisonLeeeee/GraphGallery.git
cd GraphGallery
python setup.py install
```
+ Or use Pip
```bash
pip install -U graphgallery
```

# Implementations
In detail, the following methods are currently implemented:

## Semi-supervised models
### General 

+ **ChebyNet** from *Michaël Defferrard et al*, [📝Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), *NIPS'16*. 
 [[:octocat:Official Codes]](https://github.com/mdeff/cnn_graph), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/test_ChebyNet.ipynb)
+ **GCN** from *Thomas N. Kipf et al*, [📝Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), *ICLR'17*. 
 [[:octocat:Official Codes]](https://github.com/tkipf/gcn), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/test_GCN.ipynb)
+ **GraphSAGE** from *William L. Hamilton et al*, [📝Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216), *NIPS'17*. 
 [[:octocat:Official Codes]](https://github.com/williamleif/GraphSAGE), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/test_GraphSAGE.ipynb)
+ **FastGCN** from *Jie Chen et al*, [FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling](https://arxiv.org/abs/1801.10247), *ICLR'18*. 
 [[:octocat:Official Codes]](https://github.com/matenure/FastGCN), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/test_FastGCN.ipynb)
+ **LGCN** from  *Hongyang Gao et al*, [📝Large-Scale Learnable Graph Convolutional Networks](https://arxiv.org/abs/1808.03965), *KDD'18*. 
 [[:octocat:Official Codes]](https://github.com/divelab/lgcn), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/test_LGCN.ipynb)
+ **GAT** from *Petar Veličković et al*, [📝Graph Attention Networks](https://arxiv.org/abs/1710.10903), *ICLR'18*. 
 [[:octocat:Official Codes]](https://github.com/PetarV-/GAT), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/test_GAT.ipynb)
+ **SGC** from *Felix Wu et al*, [📝Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153), *ICML'19*. 
 [[:octocat:Official Codes]](https://github.com/Tiiiger/SGC), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/test_SGC.ipynb)
+ **GWNN** from *Bingbing Xu et al*, [📝Graph Wavelet Neural Network](https://arxiv.org/abs/1904.07785), *ICLR'19*. 
 [[:octocat:Official Codes]](https://github.com/Eilene/GWNN), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/test_GWNN.ipynb)
+ **GMNN** from *Meng Qu et al*, [📝Graph Markov Neural Networks](https://arxiv.org/abs/1905.06214), *ICML'19*. 
 [[:octocat:Official Codes]](https://github.com/DeepGraphLearning/GMNN), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/test_GMNN.ipynb)
+ **ClusterGCN** from *Wei-Lin Chiang et al*, [📝Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/abs/1905.07953), *KDD'19*. 
 [[:octocat:Official Codes]](https://github.com/google-research/google-research/tree/master/cluster_gcn), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/test_ClusterGCN.ipynb)
+ **DAGNN** from *Meng Liu et al*, [📝Towards Deeper Graph Neural Networks](https://arxiv.org/abs/2007.09296), *KDD'20*. 
 [[:octocat:Official Codes]](https://github.com/mengliu1998/DeeperGNN), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/test_DAGNN.ipynb)


### Defense models
+ **RobustGCN** from *Dingyuan Zhu et al*, [📝Robust Graph Convolutional Networks Against Adversarial Attacks](https://dl.acm.org/doi/10.1145/3292500.3330851), *KDD'19*. 
 [[:octocat:Official Codes]](https://github.com/thumanlab/nrlweb/blob/master/static/assets/download/RGCN.zip), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/test_RobustGCN.ipynb)
+ **SBVAT/OBVAT** from *Zhijie Deng et al*, [📝Batch Virtual Adversarial Training for Graph Convolutional Networks](https://arxiv.org/abs/1902.09192), *ICML'19*. 
 [[:octocat:Official Codes]](https://github.com/thudzj/BVAT), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/test_SBVAT.ipynb), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/test_OBVAT.ipynb)

## Unsupervised models
+ **Deepwalk** from *Bryan Perozzi et al*, [📝DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652), *KDD'14*. 
 [[:octocat:Official Codes]](https://github.com/phanein/deepwalk), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/test_Deepwalk.ipynb)
+ **Node2vec** from *Aditya Grover et al*, [📝node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653), *KDD'16*. 
 [[:octocat:Official Codes]](https://github.com/aditya-grover/node2vec), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/test_Node2vec.ipynb)

# Quick Start
## Datasets
```python
from graphgallery.data import Planetoid
# set `verbose=False` to avoid these printed tables
data = Planetoid('cora', verbose=False)
adj, x, labels = data.graph.unpack()
idx_train, idx_val, idx_test = data.split()
# adj:  adjacency matrix: 2D Scipy sparse matrix
# x:  feature matrix: 2D Numpy array
# labels:  class labels: 1D Numpy array
# idx_train:  training indices: 1D Numpy array
# idx_val:  validation indices: 1D Numpy array
# idx_test:  testing indices: 1D Numpy array
```
currently the supported datasets are:
```python
>>> data.supported_datasets
('citeseer', 'cora', 'pubmed')
```

## Example of GCN model
```python
from graphgallery.nn.models import GCN
# adj is scipy sparse matrix, x is numpy array matrix
model = GCN(adj, x, labels, device='GPU', norm_x='l1', seed=123)
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

# other parameters like `dropouts` and `l2_norms` (if have) are the SAME.
```
+ Train your model
```python
# train with validation
>>> his = model.train(idx_train, idx_val, verbose=1, epochs=100)
# train without validation
# his = model.train(idx_train, verbose=1, epochs=100)
```
here `his` is tensorflow `Histoory` like instance (or itself).

+ Test you model
```python
loss, accuracy = model.test(idx_test)
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```
+ Display hyper-parameters

You can simply use `model.show()` to show all your `Hyper-parameters`.
Otherwise you can also use `model.show('model')` or `model.show('train')` to show your model parameters and training parameters.

NOTE: you should install texttable first.

## Visualization
NOTE: you must install [SciencePlots](https://github.com/garrettj403/SciencePlots) package for a better preview.
+ Accuracy
```python
import matplotlib.pyplot as plt
with plt.style.context(['science', 'no-latex']):
    plt.plot(his.history['acc'])
    plt.plot(his.history['val_acc'])
    plt.legend(['Train Accuracy', 'Val Accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.autoscale(tight=True)
    plt.show()    
```
![visualization](https://github.com/EdisonLeeeee/GraphGallery/blob/master/imgs/visualization_acc.png)

+ Loss
```python
import matplotlib.pyplot as plt
with plt.style.context(['science', 'no-latex']):
    plt.plot(his.history['loss'])
    plt.plot(his.history['val_loss'])
    plt.legend(['Train Loss', 'Val Loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.autoscale(tight=True)
    plt.show()    
```
![visualization](https://github.com/EdisonLeeeee/GraphGallery/blob/master/imgs/visualization_loss.png)

# More Examples
Please refer to the [examples](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples) directory.

# TODO Lists
- [ ] Add Docstrings and Documentation
- [ ] Add PyTorch models support
- [ ] Support for `graph Classification` and `link prediction` tasks
- [ ] Support for Heterogeneous graphs


# Acknowledgement
This project is motivated by [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric), [Tensorflow Geometric](https://github.com/CrawlScript/tf_geometric) and [Stellargraph](https://github.com/stellargraph/stellargraph), and the original implementations of the authors, thanks for their excellent works!

