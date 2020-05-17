<a class="toc" id="table-of-contents"></a>
# Table of Contents
+ [GraphGallery](#1)
+ [Requirements](#2)
+ [Usage](#3)
	+ [Inputs](#3-1)
	+ [GCN](#3-2)
	+ [DenceGCN](#3-3)
	+ [EdgeGCN](#3-4)
	+ [GDC](#3-5)
	+ [ChebyNet](#3-6)
	+ [FastGCN](#3-7)
	+ [GraphSAGE](#3-8)
	+ [RobustGCN](#3-9)
	+ [SGC](#3-10)
	+ [GWNN](#3-11)
	+ [GAT](#3-12)
	+ [ClusterGCN](#3-13)
	+ [SBVAT](#3-14)
	+ [OBVAT](#3-15)
	+ [GMNN](#3-16)
	+ [LGCN](#3-17)
	+ [Deepwalk](#3-18)
	+ [Node2vec](#3-19)


<a class="toc" id ="1"></a>
# GraphGallery
[🔙](#table-of-contents)

A gallery of state-of-the-arts graph neural networks.. 

Implemented with Tensorflow 2.x.

This repo aims to achieve 4 goals:
+ Similar (or higher) performance with the corresponding papers
+ Faster implementation of training and testing
+ Simple and convenient to use with high scalability
+ Easy to read source codes

<a class="toc" id ="2"></a>
# Requirements
[🔙](#table-of-contents)

+ python>=3.7
+ tensorflow>=2.1 (2.1 is recommended)
+ networkx==2.3
+ metis==0.2a4
+ scipy>=1.4.1
+ sklearn>=0.22
+ numpy>=1.18.1
+ numba>=0.48
+ gensim>=3.8.1
+ texttable

To install `metis`, just:
```bash
sudo apt-get install libmetis-dev 
pip install metis
```


<a class="toc" id ="3"></a>
# Usage
[🔙](#table-of-contents)

<a class="toc" id ="3-1"></a>
## Inputs
[🔙](#table-of-contents)

+ adj: `scipy.sparse.csr_matrix` (or `csc_matrix`) with shape (N, N)
    The input `symmetric` adjacency matrix, where `N` is the number of nodes 
    in graph.
+ features: `np.array` with shape (N, F)
    The input node feature matrix, where `F` is the dimension of node features.
+ labels: `np.array` with shape (N,)
    The ground-truth labels for all nodes in graph.
+ idx_train: `np.array`, `list`, Integer scalar or `graphgallery.NodeSequence`
    the index of nodes (or sequence) that will be used during training.    
+ idx_val: `np.array`, `list`, Integer scalar or `graphgallery, optional.NodeSequence`
    the index of nodes (or sequence) that will be used for validation. 
    (default :obj: `None`, i.e., do not use validation during training)
+ idx_test: `np.array`, `list`, Integer scalar or `graphgallery.NodeSequence`
    The index of nodes (or sequence) that will be tested.   
+ device (String, optional): 
    The device where the model is running on. You can specified `CPU` or `GPU` 
    for the model. (default: :obj: `CPU:0`, i.e., the model is running on 
    the 0-th device `CPU`)
+ seed (Positive integer, optional): 
    Used in combination with `tf.random.set_seed & np.random.seed & random.seed` 
    to create a reproducible sequence of tensors across multiple calls. 
    (default :obj: `None`, i.e., using random seed)



You can specified customized hyperparameters and training details by calling `model.build(your_args)` and `model.trian(your_args)`. 
The usuage documents will be gradually added later.

<a class="toc" id ="3-2"></a>
## GCN
[🔙](#table-of-contents)

+ [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), ICLR 2017
+ Tensorflow 1.x implementation: https://github.com/tkipf/gcn
+ Pytorch implementation: https://github.com/tkipf/pygcn

```python
from graphgallery.nn.models import GCN
model = GCN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close # clear session
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```


<a class="toc" id ="3-3"></a>
## DenceGCN
[🔙](#table-of-contents)

Dense version of `GCN`, i.e., the `adj` will be transformed to `Tensor` instead of `SparseTensor`.

```python
from graphgallery.nn.models import DenseGCN
model = DenseGCN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close # clear session
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-4"></a>
## EdgeGCN
[🔙](#table-of-contents)

Edge Convolutional version of `GCN`, using message passing framework,
i.e., using Tensor `edge index` and `edge weight` of adjacency matrix to aggregate neighbors'
message, instead of SparseTensor `adj`.

Inspired by: tf_geometric and torch_geometric
+ tf_geometric: https://github.com/CrawlScript/tf_geometric
+ torch_geometric: https://github.com/rusty1s/pytorch_geometric

```python
from graphgallery.nn.models import EdgeGCN
model = EdgeGCN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close # clear session
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-5"></a>
## GDC
[🔙](#table-of-contents)

+ [Diffusion Improves Graph Learning](https://arxiv.org/abs/1911.05485), NeurIPS 2019
+ official implementation: https://github.com/klicperajo/gdc
+ torch_geometric implementation: https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/transforms/gdc.py


```python
from graphgallery.nn.models import GCN
from graphgallery.utils import GDC

GDC_adj = GDC(adj, alpha=0.3, k=128, which='PPR')
model = GCN(GDC_adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close # clear session
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-6"></a>
## ChebyNet
[🔙](#table-of-contents)

+ [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), NeurIPS 2016
+ Tensorflow 1.x implementation: https://github.com/mdeff/cnn_graph, https://github.com/tkipf/gcn
+ Keras implementation: https://github.com/aclyde11/ChebyGCN

```python
from graphgallery.nn.models import ChebyNet
model = ChebyNet(adj, features, labels, order=2, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close # clear session
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-7"></a>
## FastGCN
[🔙](#table-of-contents)

+ [FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling](https://arxiv.org/abs/1801.10247), ICLR 2018
+ Tensorflow 1.x implementation: https://github.com/matenure/FastGCN

```python
from graphgallery.nn.models import FastGCN
model = FastGCN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close # clear session
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-8"></a>
## GraphSAGE
[🔙](#table-of-contents)

+ [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216), NeurIPS 2017
+ Tensorflow 1.x implementation: https://github.com/williamleif/GraphSAGE
+ Pytorch implementation: https://github.com/williamleif/graphsage-simple/

```python
from graphgallery.nn.models import GraphSAGE
model = GraphSAGE(adj, features, labels, n_samples=[10, 5], device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100, restore_best=False, validation=False)
loss, accuracy = model.test(idx_test)
model.close # clear session
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-9"></a>
## RobustGCN
[🔙](#table-of-contents)

+ [Robust Graph Convolutional Networks Against Adversarial Attacks](https://dl.acm.org/doi/10.1145/3292500.3330851), KDD 2019
+ Tensorflow 1.x implementation: https://github.com/thumanlab/nrlweb/blob/master/static/assets/download/RGCN.zip

```python
from graphgallery.nn.models import RobustGCN
model = RobustGCN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close # clear session
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-10"></a>
## SGC
[🔙](#table-of-contents)

+ [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153), ICML 2019
+ Pytorch implementation: https://github.com/Tiiiger/SGC
        
```python
from graphgallery.nn.models import SGC
model = SGC(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close # clear session
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-11"></a>
## GWNN
[🔙](#table-of-contents)

+ [Graph Wavelet Neural Network](https://arxiv.org/abs/1904.07785), ICLR 2019
+ Tensorflow 1.x implementation: https://github.com/Eilene/GWNN
+ Pytorch implementation: https://github.com/benedekrozemberczki/GraphWaveletNeuralNetwork
        
```python
from graphgallery.nn.models import GWNN
model = GWNN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close # clear session
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-12"></a>
## GAT
[🔙](#table-of-contents)

+ [Graph Attention Networks](https://arxiv.org/abs/1710.10903), ICLR 2018
+ Tensorflow 1.x implementation: https://github.com/PetarV-/GAT
+ Pytorch implementation: https://github.com/Diego999/pyGAT
+ Keras implementation: https://github.com/danielegrattarola/keras-gat
        
```python
from graphgallery.nn.models import GAT
model = GAT(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=200)
loss, accuracy = model.test(idx_test)
model.close # clear session
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-13"></a>
## ClusterGCN
[🔙](#table-of-contents)

+ [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/abs/1905.07953), KDD 2019
+ Tensorflow 1.x implementation: https://github.com/google-research/google-research/tree/master/cluster_gcn
+ Pytorch implementation: https://github.com/benedekrozemberczki/ClusterGCN

```python
from graphgallery.nn.models import ClusterGCN
model = ClusterGCN(adj, features, labels, n_cluster=10, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close # clear session
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-14"></a>
## SBVAT
[🔙](#table-of-contents)

+ [Batch Virtual Adversarial Training for Graph Convolutional Networks](https://arxiv.org/abs/1902.09192), ICML 2019
+ Tensorflow 1.x implementation: https://github.com/thudzj/BVAT

```python
from graphgallery.nn.models import SBVAT
model = GMNN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close # clear session
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```


<a class="toc" id ="3-15"></a>
## OBVAT
[🔙](#table-of-contents)

+ [Batch Virtual Adversarial Training for Graph Convolutional Networks](https://arxiv.org/abs/1902.09192), ICML 2019
+ Tensorflow 1.x implementation: https://github.com/thudzj/BVAT

```python
from graphgallery.nn.models import OBVAT
model = OBVAT(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close # clear session
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```


<a class="toc" id ="3-16"></a>
## GMNN
[🔙](#table-of-contents)

+ [Graph Markov Neural Networks](https://arxiv.org/abs/1905.06214), ICML 2019
+ Pytorch implementation: https://github.com/DeepGraphLearning/GMNN


```python
from graphgallery.nn.models import GMNN
model = GMNN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close # clear session
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-17"></a>
## LGCN
[🔙](#table-of-contents)

+ [Large-Scale Learnable Graph Convolutional Networks](https://arxiv.org/abs/1808.03965), KDD 2018
+ Tensorflow 1.x implementation: https://github.com/divelab/lgcn

```python
from graphgallery.nn.models import LGCN
model = GMNN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close # clear session
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-18"></a>
## Deepwalk
[🔙](#table-of-contents)

+ [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652), KDD 2014
+ Implementation: https://github.com/phanein/deepwalk

```python
from graphgallery.nn.models import Deepwalk
model = Deepwalk(adj, features, labels)
model.build()
model.train(idx_train)
accuracy = model.test(idx_test)
print(f'Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-19"></a>
## Node2vec
[🔙](#table-of-contents)

+ [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653), KDD 2016
+ Implementation: https://github.com/aditya-grover/node2vec
+ Cpp implementation: https://github.com/snap-stanford/snap/tree/master/examples/node2vec

```python
from graphgallery.nn.models import Node2vec
model = Node2vec(adj, features, labels)
model.build()
model.train(idx_train)
accuracy = model.test(idx_test)
print(f'Test accuracy {accuracy:.2%}')
```
