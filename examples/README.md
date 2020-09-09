# Implementations
In detail, the following methods are currently implemented:

## Semi-supervised models
### General 

+ **ChebyNet** from *Michaël Defferrard et al*, [📝Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), *NIPS'16*. 
 [[🌋 TensorFlow]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_ChebyNet.ipynb)
+ **GCN** from *Thomas N. Kipf et al*, [📝Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), *ICLR'17*. 
 [[🌋 TensorFlow]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_GCN.ipynb), [[🔥 PyTorch]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/PyTorch/test_GCN.ipynb)
+ **GraphSAGE** from *William L. Hamilton et al*, [📝Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216), *NIPS'17*. 
 [[🌋 TensorFlow]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_GraphSAGE.ipynb)
+ **FastGCN** from *Jie Chen et al*, [FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling](https://arxiv.org/abs/1801.10247), *ICLR'18*. 
 [[🌋 TensorFlow]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_FastGCN.ipynb)
+ **LGCN** from  *Hongyang Gao et al*, [📝Large-Scale Learnable Graph Convolutional Networks](https://arxiv.org/abs/1808.03965), *KDD'18*. 
 [[🌋 TensorFlow]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_LGCN.ipynb)
+ **GAT** from *Petar Veličković et al*, [📝Graph Attention Networks](https://arxiv.org/abs/1710.10903), *ICLR'18*. 
 ), [[🌋 TensorFlow]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_GAT.ipynb)
+ **SGC** from *Felix Wu et al*, [📝Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153), *ICML'19*. 
 [[🌋 TensorFlow]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_SGC.ipynb)
+ **GWNN** from *Bingbing Xu et al*, [📝Graph Wavelet Neural Network](https://arxiv.org/abs/1904.07785), *ICLR'19*. 
 [[🌋 TensorFlow]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_GWNN.ipynb)
+ **GMNN** from *Meng Qu et al*, [📝Graph Markov Neural Networks](https://arxiv.org/abs/1905.06214), *ICML'19*. 
 [[🌋 TensorFlow]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_GMNN.ipynb)
+ **ClusterGCN** from *Wei-Lin Chiang et al*, [📝Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/abs/1905.07953), *KDD'19*. 
[[🌋 TensorFlow]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_ClusterGCN.ipynb)
+ **DAGNN** from *Meng Liu et al*, [📝Towards Deeper Graph Neural Networks](https://arxiv.org/abs/2007.09296), *KDD'20*. 
 [[🌋 TensorFlow]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_DAGNN.ipynb)


### Defense models
+ **RobustGCN** from *Dingyuan Zhu et al*, [📝Robust Graph Convolutional Networks Against Adversarial Attacks](https://dl.acm.org/doi/10.1145/3292500.3330851), *KDD'19*. 
  [[🌋 TensorFlow]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_RobustGCN.ipynb)
+ **SBVAT/OBVAT** from *Zhijie Deng et al*, [📝Batch Virtual Adversarial Training for Graph Convolutional Networks](https://arxiv.org/abs/1902.09192), *ICML'19*. 
 [[🌋 TensorFlow]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_SBVAT.ipynb), [[🌋 TensorFlow]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_OBVAT.ipynb)

## Unsupervised models
+ **Deepwalk** from *Bryan Perozzi et al*, [📝DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652), *KDD'14*. 
 [[🌋 TensorFlow]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_Deepwalk.ipynb)
+ **Node2vec** from *Aditya Grover et al*, [📝node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653), *KDD'16*. 
 de2vec), [[🌋 TensorFlow]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/TensorFlow/test_Node2vec.ipynb)