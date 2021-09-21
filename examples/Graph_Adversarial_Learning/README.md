# Implementations

In detail, the following methods are currently implemented:

## Attack

### Targeted Attack

#### Common

- **RAND**: The simplest attack method.
  [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning/Targeted/Poisoning/Common/RAND.py)
- **DICE**, from _Marcin Waniek et al_, [📝Hiding Individuals and Communities in a Social Network](https://arxiv.org/abs/1608.00375), _Nature Human Behavior 16_.
  [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning/Targeted/Poisoning/Common/DICE.py)
- **Nettack**, from _Daniel Zügner et al._, [📝Adversarial Attacks on Neural Networks for Graph Data](https://arxiv.org/abs/1805.07984), _KDD'18_.
  [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning/Targeted/Poisoning/Common/Nettack.py)
- **GF-Attack**, from _Heng Chang et al_, [📝A Restricted Black-box Adversarial Framework Towards Attacking Graph Embedding Models](https://arxiv.org/abs/1908.01297), _AAAI'20_.
  [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning/Targeted/Poisoning/Common/GFA.py)

#### TensorFlow

- **FGSM**, from _Ian J. Goodfellow et al._, [📝Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), _ICLR'15_.
  [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning/Targeted/Poisoning/TensorFlow/FGSM.py)
- **IG**, from _Huijun Wu et al._, [📝Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/abs/1903.01610), _IJCAI'19_.
  [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning/Targeted/Poisoning/TensorFlow/IG.py)
- **IGA**, from _Jinyin Chen et al._, [📝Link Prediction Adversarial Attack Via Iterative Gradient Attack](https://ieeexplore.ieee.org/abstract/document/9141291), _IEEE Trans 20_.
  [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning/Targeted/Poisoning/TensorFlow/IGA.py)
- **SGA**, from _Jintang Li et al._ [📝 Adversarial Attack on Large Scale Graph](https://arxiv.org/abs/2009.03488)
  [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning/Targeted/Poisoning/TensorFlow/SGA.py)

#### PyTorch
- **SGA**, from _Jintang Li et al._ [📝 Adversarial Attack on Large Scale Graph](https://arxiv.org/abs/2009.03488)
  [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning/Targeted/Poisoning/PyTorch/SGA.py)

### Untargeted Attack

#### Common

- **RAND**: The simplest attack method.
  [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning/Untargeted/Poisoning/Common/RAND.py)
- **DICE**, from _Marcin Waniek et al_, [📝Hiding Individuals and Communities in a Social Network](https://arxiv.org/abs/1608.00375), _Nature Human Behavior 16_.
  [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning/Untargeted/Poisoning/Common/DICE.py)
- **Degree**, **Node Embedding Attack**, from _Aleksandar Bojchevski et al._, [📝Adversarial Attacks on Node Embeddings via Graph Poisoning](https://arxiv.org/abs/1809.01093), _ICLR'19_.
  [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning/Untargeted/Poisoning/Common/Degree.py), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Untargeted/Poisoning/Common/node_embedding_attack.py)

#### TensorFlow

- **FGSM**, from _Ian J. Goodfellow et al._, [📝Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), _ICLR'15_.
  [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning/Untargeted/Poisoning/TensorFlow/FGSM.py)
- **Metattack**, **MetaApprox**, from _Daniel Zügner et al._, [📝Adversarial Attacks on Graph Neural Networks via Meta Learning](https://arxiv.org/abs/1902.08412), _ICLR'19_.
  [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning/Untargeted/Poisoning/TensorFlow/Metattack.py), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning/Untargeted/Poisoning/TensorFlow/MetaApprox.py)
- **PGD**, **MinMax**, from _Kaidi Xu et al._, [📝Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](https://arxiv.org/abs/1906.04214), _IJCAI'19_.
  [[🌈 Poisoning Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Untargeted/Poisoning/TensorFlow/PGD.py), [[🌈 Poisoning Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Untargeted/Poisoning/TensorFlow/MinMax.py), [[🌈 Evasion Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Untargeted/Evasion/TensorFlow/PGD.py)

#### PyTorch

- **Metattack**, **MetaApprox**, from _Daniel Zügner et al._, [📝Adversarial Attacks on Graph Neural Networks via Meta Learning](https://arxiv.org/abs/1902.08412), _ICLR'19_.
  [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning/Untargeted/Poisoning/PyTorch/Metattack.py), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphGallery/blob/master/examples/Graph_Adversarial_Learning/Untargeted/Poisoning/PyTorch/MetaApprox.py)

<!-- ## Defense
+ **JaccardDetection**, **CosinDetection**, from *Huijun Wu et al.*, [📝Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/abs/1903.01610), *IJCAI'19*.
 [[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Defense/detection.py)
+ **Adversarial Tranining**, from *Kaidi Xu et al.*, [📝Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](https://arxiv.org/abs/1906.04214), *IJCAI'19*.
+ **SVD**, from *Negin Entezari et al.*, [📝All You Need Is Low (Rank): Defending Against Adversarial Attacks on Graphs](https://dl.acm.org/doi/abs/10.1145/3336191.3371789), *WSDM'20*.
 [[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Defense/svd.py)
+ **RGCN**, from *Dingyuan Zhu et al.*, [Robust Graph Convolutional Networks Against Adversarial Attacks](http://pengcui.thumedialab.com/papers/RGCN.pdf), *KDD'19*.
 [[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Defense/RGCN.py) -->

More details of the official papers and codes can be found in [Awesome Graph Adversarial Learning](https://github.com/gitgiter/Graph-Adversarial-Learning).
