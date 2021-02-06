- [🤖 Implementations](#-implementations)
  - [Semi-supervised models](#semi-supervised-models)
    - [General models](#general-models)
    - [Defense models (for Graph Adversarial Learning)](#defense-models-for-graph-adversarial-learning)
      - [Robust Optimization](#robust-optimization)
      - [Graph Purification](#graph-purification)
  - [Unsupervised models](#unsupervised-models)

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
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/ChebyNet.py"> [:octocat:TensorFLow]</a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/ChebyNet.py"> [🔥PyTorch] </a>

</details>

<!-- 2 -->

<details>
<summary>
<b>GCN</b> from <i>Thomas N. Kipf et al</i>,
<a href="https://arxiv.org/abs/1609.02907"> 📝<i>Semi-Supervised Classification with Graph Convolutional Networks (ICLR'17)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/GCN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GCN.py"> [🔥PyTorch] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyG/GCN.py"> [🔥PyG] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/DGL-PyTorch/GCN.py"> [🔥DGL-PyTorch] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/DGL-TensorFlow/GCN.py"> [:octocat:DGL-TensorFlow] </a>
</details>

<!-- 3 -->
<details>
<summary>
<b>GraphSAGE</b> from <i>William L. Hamilton et al</i>,
<a href="https://arxiv.org/abs/1706.02216"> 📝<i>Inductive Representation Learning on Large Graphs (NeurIPS'17)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/GraphSAGE.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GraphSAGE.py"> [🔥PyTorch] </a>
</details>

<!-- 4 -->
<details>
<summary>
<b>FastGCN</b> from <i>Jie Chen et al</i>,
<a href="https://arxiv.org/abs/1801.10247"> 📝<i>FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling (ICLR'18)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/FastGCN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/FastGCN.py"> [🔥PyTorch] </a>
</details>

<!-- 5 -->
<details>
<summary>
<b>LGCN</b> from <i>Hongyang Gao et al</i>,
<a href="https://arxiv.org/abs/1808.03965"> 📝<i>Large-Scale Learnable Graph Convolutional Networks (KDD'18)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/LGCN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/LGCN.py"> [🔥PyTorch] </a>
</details>

<!-- 6 -->
<details>
<summary>
<b>GAT</b> from <i>Petar Veličković et al</i>,
<a href="https://arxiv.org/abs/1710.10903"> 📝<i>Graph Attention Networks (ICLR'18)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/GAT.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GAT.py"> [🔥PyTorch] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyG/GAT.py"> [🔥PyG] </a>
</details>

<!-- 7 -->
<details>
<summary>
<b>SGC</b> from <i>Felix Wu et al</i>,
<a href="https://arxiv.org/abs/1902.07153"> 📝<i>Simplifying Graph Convolutional Networks (ICLR'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/SGC.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/SGC.py"> [🔥PyTorch] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyG/SGC.py"> [🔥PyG] </a>
</details>

<!-- 8 -->
<details>
<summary>
<b>GWNN</b> from <i>Bingbing Xu et al</i>,
<a href="https://arxiv.org/abs/1904.07785"> 📝<i>Graph Wavelet Neural Network (ICLR'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/GWNN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GWNN.py"> [🔥PyTorch] </a>
</details>

<!-- 9 -->
<details>
<summary>
<b>GMNN</b> from <i>Meng Qu et al</i>,
<a href="https://arxiv.org/abs/1905.06214"> 📝<i>Graph Attention Networks (ICLR'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/GMNN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GMNN.py"> [🔥PyTorch] </a>
</details>

<!-- 10 -->
<details>
<summary>
<b>ClusterGCN</b> from <i>Wei-Lin Chiang et al</i>,
<a href="https://arxiv.org/abs/1905.07953"> 📝<i>Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks (KDD'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/ClusterGCN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/ClusterGCN.py"> [🔥PyTorch] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyG/ClusterGCN.py"> [🔥PyG] </a>
</details>

<!-- 11 -->
<details>
<summary>
<b>DAGNN</b> from <i>Meng Liu et al</i>,
<a href="https://arxiv.org/abs/2007.09296"> 📝<i>Towards Deeper Graph Neural Networks (KDD'20)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/DAGNN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/DAGNN.py"> [🔥PyTorch] </a>
</details>

<!-- 12 -->
<details>
<summary>
<b>GDC</b> from <i>Johannes Klicpera et al</i>,
<a href="https://www.in.tum.de/daml/gdc/"> 📝<i>Diffusion Improves Graph Learning (NeurIPS'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/GCN-GDC.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GCN-GDC.py"> [🔥PyTorch] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyG/GCN-GDC.py"> [🔥PyG] </a>
</details>

<!-- 13 -->

<details>
<summary>
<b>GCN</b> from <i>Du et al</i>,
<a href="https://arxiv.org/abs/1710.10370"> 📝<i>Topology Adaptive Graph Convolutional Networks (arxiv'17)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/TAGCN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/TAGCN.py"> [🔥PyTorch] </a>
</details>

### Defense models (for Graph Adversarial Learning)

#### Robust Optimization

<!-- 1 -->
<details>
<summary>
<b>RobustGCN</b> from <i>Petar Veličković et al</i>,
<a href="https://dl.acm.org/doi/10.1145/3292500.3330851"> 📝<i>Robust Graph Convolutional Networks Against Adversarial Attacks (KDD'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/RobustGCN.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/RobustGCN.py"> [🔥PyTorch] </a>
</details>

<!-- 2 -->
<details>
<summary>
<b>SBVAT</b> from <i>Zhijie Deng et al</i>,
<a href="https://arxiv.org/abs/1902.09192"> 📝<i>Batch Virtual Adversarial Training for Graph Convolutional Networks (ICML'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/SBVAT.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/SBVAT.py"> [🔥PyTorch] </a>
</details>

<!-- 3 -->
<details>
<summary>
<b>OBVAT</b> from <i>Zhijie Deng et al</i>,
<a href="https://arxiv.org/abs/1902.09192"> 📝<i>Batch Virtual Adversarial Training for Graph Convolutional Networks (ICML'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/OBVAT.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/OBVAT.py"> [🔥PyTorch] </a>
</details>

<!-- 4 -->
<details>
<summary>
<b>SimPGCN</b> from <i>Wei Jin et al</i>,
<a href="https://arxiv.org/abs/2011.09643"> 📝<i>Node Similarity Preserving Graph Convolutional Networks (WSDM'21)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/SimPGCN.py"> [🔥PyTorch] </a>
</details>

#### Graph Purification

The graph purification methods are universal for all models, just specify:

```python
graph_transform="purification_method"
```

so, here we only give the examples of `GCN` with purification methods, other models should work.

<!-- 1 -->
<details>
<summary>
<b>GCN-Jaccard</b> from <i>Huijun Wu et al</i>,
<a href="https://arxiv.org/abs/1903.01610"> 📝<i>Adversarial Examples on Graph Data: Deep Insights into Attack and Defense (IJCAI'19)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/GCN-Jaccard.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GCN-Jaccard.py"> [🔥PyTorch] </a>
</details>

<!-- 2 -->
<details>
<summary>
<b>GCN-SVD</b> from <i>Negin Entezari et al</i>,
<a href="https://dl.acm.org/doi/abs/10.1145/3336191.3371789"> 📝<i>All You Need Is Low (Rank): Defending Against Adversarial Attacks on Graphs (WSDM'20)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/TensorFlow/GCN-SVD.py"> [:octocat:TensorFLow] </a>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/PyTorch/GCN-SVD.py"> [🔥PyTorch] </a>
</details>

## Unsupervised models

<!-- 1 -->
<details>
<summary>
<b>Deepwalk</b> from <i>Zhijie Deng et al</i>,
<a href="https://arxiv.org/abs/1403.6652"> 📝<i>DeepWalk: Online Learning of Social Representations (KDD'14)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/Common/Deepwalk.py"> [Example] </a>
</details>

<!-- 2 -->
<details>
<summary>
<b>Node2vec</b> from <i>Zhijie Deng et al</i>,
<a href="https://arxiv.org/abs/1607.00653"> 📝<i>node2vec: Scalable Feature Learning for Networks (KDD'16)</i> </a>
</summary>
<a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Neural_Networks/Common/Node2vec.py"> [Example] </a>
</details>
