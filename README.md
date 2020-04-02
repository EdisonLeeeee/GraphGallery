# GraphGallery

TensorFlow 2 implementation of state-of-the-arts deep learning graph models.

To achieve Four goals:
+ Similar accuracy with the proposed paper
+ Faster implementation of training and testing
+ Simple and convenient to use with higher flexibility
+ Easy to read source code

# Requirements

+ python>=3.7
+ tensorflow>=2.1
+ networkx==2.3
+ metis==0.2a4
+ scipy>=1.4.1
+ sklearn>=0.22
+ numpy>=1.18.1
+ numba>=0.48
+ gensim>=3.8.1

# Usage

## Inputs

+ `adj`: scipy.sparse.csr_matrix (or csr_matrix), shape [N, N], sparse form of adjacency matrix. 
    where `N` means number of nodes in the graph.
+ `features`: np.ndarray shape [N, F], dense form of feature matrix. where `F` means dims of features.
+ `labels`: np.ndarray shape [N,], ground-truth labels of all nodes.
+ `idx_train`, `idx_val`, `idx_test`: np.ndarray, the seperated indices of training, validation and test nodes. where `len(idx_train) + len(idx_val) + len(idx_test) = N`

You can specified hyperparameter and training details by call `model.build(your_args)` and `model.trian(your_args)`. The usuage documents will be gradually added later.

## GCN

```python
from graphgallery.nn.models import GCN
model = GCN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```


## DenceGCN

```python
from graphgallery.nn.models import DenseGCN
model = DenseGCN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```



## ChebyGCN

```python
from graphgallery.nn.models import ChebyGCN
model = ChebyGCN(adj, features, labels, order=2, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

## FastGCN

```python
from graphgallery.nn.models import FastGCN
model = FastGCN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

## GraphSAGE

```python
from graphgallery.nn.models import GraphSAGE
model = GraphSAGE(adj, features, labels, n_samples=[10, 5], device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100, restore_best=False, validation=False)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

## RobustGCN

```python
from graphgallery.nn.models import RobustGCN
model = RobustGCN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

## SGC

```python
from graphgallery.nn.models import SGC
model = SGC(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

## GWNN

```python
from graphgallery.nn.models import GWNN
model = GWNN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

## GAT

```python
from graphgallery.nn.models import GAT
model = GAT(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=200)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

## ClusterGCN

```python
from graphgallery.nn.models import ClusterGCN
model = ClusterGCN(Data.adj, Data.features, data.labels, n_cluster=10, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

## SBVAT
```python
from graphgallery.nn.models import SBVAT
model = GMNN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```


## OBVAT
```python
from graphgallery.nn.models import OBVAT
model = OBVAT(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```


## GMNN
```python
from graphgallery.nn.models import GMNN
model = GMNN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

## LGCN
```python
from graphgallery.nn.models import LGCN
model = GMNN(adj, features, labels, device='CPU', seed=123)
model.build()
his = model.train(idx_train, idx_val, verbose=True, epochs=100)
loss, accuracy = model.test(idx_test)
model.close
print(f'Test loss {loss:.5}, Test accuracy {accuracy:.2%}')
```

## Deepwalk

```python
from graphgallery.nn.models import Deepwalk
model = Deepwalk(adj, features, labels)
model.build()
model.train(idx_train)
accuracy = model.test(idx_test)
print(f'Test accuracy {accuracy:.2%}')
```

## Node2vec

```python
from graphgallery.nn.models import Node2vec
model = Node2vec(adj, features, labels)
model.build()
model.train(idx_train)
accuracy = model.test(idx_test)
print(f'Test accuracy {accuracy:.2%}')
```

