<a class="toc" id="table-of-contents"></a>
# Table of Contents
+ [GraphGallery](#1)
+ [Requirements](#2)
+ [Usage](#3)
	+ [Inputs](#3-1)
	+ [GCN](#3-2)
	+ [DenceGCN](#3-3)
	+ [ChebyGCN](#3-4)
	+ [FastGCN](#3-5)
	+ [GraphSAGE](#3-6)
	+ [RobustGCN](#3-7)
	+ [SGC](#3-8)
	+ [GWNN](#3-9)
	+ [GAT](#3-10)
	+ [ClusterGCN](#3-11)
	+ [SBVAT](#3-12)
	+ [OBVAT](#3-13)
	+ [GMNN](#3-14)
	+ [LGCN](#3-15)
	+ [Deepwalk](#3-16)
	+ [Node2vec](#3-17)


# GraphGallery
[Back to TOC](#table-of-contents)


A gallery of state-of-the-arts deep learning graph models. 

Implemented with Tensorflow 2.x.

This repo aims to achieve Four goals:
+ Similar accuracy with the proposed paper
+ Faster implementation of training and testing
+ Simple and convenient to use with high scalability.
+ Easy to read source code

<a class="toc" id ="2"></a>
# Requirements
[Back to TOC](#table-of-contents)


+ python>=3.7
+ tensorflow>=2.1
+ networkx==2.3
+ metis==0.2a4
+ scipy>=1.4.1
+ sklearn>=0.22
+ numpy>=1.18.1
+ numba>=0.48
+ gensim>=3.8.1

<a class="toc" id ="3"></a>
# Usage
[Back to TOC](#table-of-contents)


<a class="toc" id ="3-1"></a>
## Inputs

+ `adj`: scipy.sparse.csr_matrix (or csr_matrix), shape [N, N], sparse form of adjacency matrix. 
    where `N` means the number of nodes in the graph.
+ `features`: np.ndarray shape [N, F], dense form of feature matrix. where `F` means the dimension of features.
+ `labels`: np.ndarray shape [N,], ground-truth labels of all nodes.
+ `idx_train`, `idx_val`, `idx_test`: np.ndarray, the seperated indices for training, validation and testing.
+ `device`: `CPU` or `GPU`.
+ `seed`: Positive interger.

You can specified customized hyperparameters and training details by call `model.build(your_args)` and `model.trian(your_args)`. 
The usuage documents will be gradually added later.

<a class="toc" id ="3-2"></a>
## GCN

[Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
Tensorflow 1.x implementation: https://github.com/tkipf/gcn
Pytorch implementation: https://github.com/tkipf/pygcn

```python
from graphgallery.nn.models import GCN
model = GCN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```


<a class="toc" id ="3-3"></a>
## DenceGCN
Dense version of `GCN`, i.e., the `adj` will be transformed to `Tensor` instead of `SparseTensor`.

```python
from graphgallery.nn.models import DenseGCN
model = DenseGCN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```



<a class="toc" id ="3-4"></a>
## ChebyGCN

[Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375)
Tensorflow 1.x implementation: https://github.com/mdeff/cnn_graph, https://github.com/tkipf/gcn
Keras implementation: https://github.com/aclyde11/ChebyGCN

```python
from graphgallery.nn.models import ChebyGCN
model = ChebyGCN(adj, features, labels, order=2, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-5"></a>
## FastGCN

[FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling](https://arxiv.org/abs/1801.10247)
Tensorflow 1.x implementation: https://github.com/matenure/FastGCN

```python
from graphgallery.nn.models import FastGCN
model = FastGCN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-6"></a>
## GraphSAGE

[Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
Tensorflow 1.x implementation: https://github.com/williamleif/GraphSAGE
Pytorch implementation: https://github.com/williamleif/graphsage-simple/

```python
from graphgallery.nn.models import GraphSAGE
model = GraphSAGE(adj, features, labels, n_samples=[10, 5], device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100, restore_best=False, validation=False)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-7"></a>
## RobustGCN

[Robust Graph Convolutional Networks Against Adversarial Attacks](https://dl.acm.org/doi/10.1145/3292500.3330851)
Tensorflow 1.x implementation: https://github.com/thumanlab/nrlweb/blob/master/static/assets/download/RGCN.zip

```python
from graphgallery.nn.models import RobustGCN
model = RobustGCN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-8"></a>
## SGC

[Simplifying Graph Convolutional Networks](https://arxiv.org/pdf/1902.07153)
Pytorch implementation: https://github.com/Tiiiger/SGC
        
```python
from graphgallery.nn.models import SGC
model = SGC(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-9"></a>
## GWNN

[Graph Wavelet Neural Network](https://arxiv.org/abs/1904.07785)
Tensorflow 1.x implementation: https://github.com/Eilene/GWNN
Pytorch implementation: https://github.com/benedekrozemberczki/GraphWaveletNeuralNetwork
        
```python
from graphgallery.nn.models import GWNN
model = GWNN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-10"></a>
## GAT

[Graph Attention Networks](https://arxiv.org/abs/1710.10903)
Tensorflow 1.x implementation: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
Keras implementation: https://github.com/danielegrattarola/keras-gat
        
```python
from graphgallery.nn.models import GAT
model = GAT(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=200)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-11"></a>
## ClusterGCN

[Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/abs/1905.07953)
Tensorflow 1.x implementation: https://github.com/google-research/google-research/tree/master/cluster_gcn
Pytorch implementation: https://github.com/benedekrozemberczki/ClusterGCN

```python
from graphgallery.nn.models import ClusterGCN
model = ClusterGCN(Data.adj, Data.features, data.labels, n_cluster=10, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-12"></a>
## SBVAT

[Batch Virtual Adversarial Training for Graph Convolutional Networks](https://arxiv.org/pdf/1902.09192)
Tensorflow 1.x implementation: https://github.com/thudzj/BVAT

```python
from graphgallery.nn.models import SBVAT
model = GMNN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```


<a class="toc" id ="3-13"></a>
## OBVAT

[Batch Virtual Adversarial Training for Graph Convolutional Networks](https://arxiv.org/pdf/1902.09192)
Tensorflow 1.x implementation: https://github.com/thudzj/BVAT

```python
from graphgallery.nn.models import OBVAT
model = OBVAT(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```


<a class="toc" id ="3-14"></a>
## GMNN

[Graph Markov Neural Networks](https://arxiv.org/abs/1905.06214)
Pytorch implementation: https://github.com/DeepGraphLearning/GMNN


```python
from graphgallery.nn.models import GMNN
model = GMNN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-15"></a>
## LGCN

[Large-Scale Learnable Graph Convolutional Networks](https://arxiv.org/abs/1808.03965)
Tensorflow 1.x implementation: https://github.com/divelab/lgcn

```python
from graphgallery.nn.models import LGCN
model = GMNN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-16"></a>
## Deepwalk

[DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)
Implementation: https://github.com/phanein/deepwalk

```python
from graphgallery.nn.models import Deepwalk
model = Deepwalk(adj, features, labels)
model.build()
model.train(idx_train)
accuracy = model.test(idx_test)
print(f'Test accuracy {accuracy:.2%}')
```

<a class="toc" id ="3-17"></a>
## Node2vec

[node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653)
Implementation: https://github.com/aditya-grover/node2vec
Cpp implementation: https://github.com/snap-stanford/snap/tree/master/examples/node2vec

```python
from graphgallery.nn.models import Node2vec
model = Node2vec(adj, features, labels)
model.build()
model.train(idx_train)
accuracy = model.test(idx_test)
print(f'Test accuracy {accuracy:.2%}')
```