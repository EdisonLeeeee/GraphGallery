# Implementations
In detail, the following methods are currently implemented:

## Attack
### Targeted Attack
+ **RAND**: The simplest attack method.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/targeted/Poisoning/RAND.py)
+ **FGSM**, from *Ian J. Goodfellow et al.*, [📝Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), *ICLR'15*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/targeted/Poisoning/FGSM.py)
+ **DICE**, from *Marcin Waniek et al*, [📝Hiding Individuals and Communities in a Social Network](https://arxiv.org/abs/1608.00375), *Nature Human Behavior 16*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/targeted/Poisoning/DICE.py)
+ **Nettack**, from *Daniel Zügner et al.*, [📝Adversarial Attacks on Neural Networks for Graph Data](https://arxiv.org/abs/1805.07984), *KDD'18*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/targeted/Poisoning/Nettack.py)
+ **IG**, from *Huijun Wu et al.*, [📝Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/abs/1903.01610), *IJCAI'19*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/targeted/Poisoning/IG.py)
+ **GF-Attack**, from *Heng Chang et al*, [📝A Restricted Black-box Adversarial Framework Towards Attacking Graph Embedding Models](https://arxiv.org/abs/1908.01297), *AAAI'20*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/targeted/Poisoning/GFA.py)
+ **IGA**, from *Jinyin Chen et al.*, [📝Link Prediction Adversarial Attack Via Iterative Gradient Attack](https://ieeexplore.ieee.org/abstract/document/9141291), *IEEE Trans 20*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/targeted/Poisoning/IGA.py)
+ **SGA**, from.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/targeted/Poisoning/SGA.py)

### Untargeted Attack
+ **RAND**: The simplest attack method.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Untargeted/Poisoning//RAND.py)
+ **FGSM**, from *Ian J. Goodfellow et al.*, [📝Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), *ICLR'15*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Untargeted/Poisoning//FGSM.py)
+ **DICE**, from *Marcin Waniek et al*, [📝Hiding Individuals and Communities in a Social Network](https://arxiv.org/abs/1608.00375), *Nature Human Behavior 16*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Untargeted/Poisoning//DICE.py)
+ **Metattack**, **MetaApprox**, from *Daniel Zügner et al.*, [📝Adversarial Attacks on Graph Neural Networks via Meta Learning](https://arxiv.org/abs/1902.08412), *ICLR'19*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Untargeted/Poisoning//Metattack.py), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Untargeted/Poisoning//MetaApprox.py)
+ **Degree**, **Node Embedding Attack**, from *Aleksandar Bojchevski et al.*, [📝Adversarial Attacks on Node Embeddings via Graph Poisoning](https://arxiv.org/abs/1809.01093), *ICLR'19*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Untargeted/Poisoning//Degree.py), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Untargeted/Poisoning//node_embedding_attack.py)
+ **PGD**, **MinMax**, from *Kaidi Xu et al.*, [📝Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](https://arxiv.org/abs/1906.04214), *IJCAI'19*.
[[🌈 Poisoning Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Untargeted/Poisoning//PGD.py), [[🌈 Poisoning Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Untargeted/Poisoning//MinMax.py), [[🌈 Evasion Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Untargeted/Poisoning/Evasion/PGD.py)

<!-- ## Defense
+ **JaccardDetection**, **CosinDetection**, from *Huijun Wu et al.*, [📝Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/abs/1903.01610), *IJCAI'19*.
 [[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Defense/detection.py)
+ **Adversarial Tranining**, from *Kaidi Xu et al.*, [📝Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](https://arxiv.org/abs/1906.04214), *IJCAI'19*.
+ **SVD**, from *Negin Entezari et al.*, [📝All You Need Is Low (Rank): Defending Against Adversarial Attacks on Graphs](https://dl.acm.org/doi/abs/10.1145/3336191.3371789), *WSDM'20*.
 [[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Defense/svd.py)
+ **RGCN**, from *Dingyuan Zhu et al.*, [Robust Graph Convolutional Networks Against Adversarial Attacks](http://pengcui.thumedialab.com/papers/RGCN.pdf), *KDD'19*.
 [[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Defense/RGCN.py) -->

More details of the official papers and codes can be found in [Awesome Graph Adversarial Learning](https://github.com/gitgiter/Graph-Adversarial-Learning).