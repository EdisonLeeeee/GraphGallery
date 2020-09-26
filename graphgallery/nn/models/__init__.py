# Base model

from graphgallery.nn.models.base_model import BaseModel
from graphgallery.nn.models.torch_keras_model import TorchKerasModel

from graphgallery import backend

_BACKEND = backend()

# (semi-)supervised model
from graphgallery.nn.models.semisupervised.semi_supervised_model import SemiSupervisedModel
if _BACKEND .kind == "T":

    from graphgallery.nn.models.semisupervised.TF.gcn import GCN
    from graphgallery.nn.models.semisupervised.TF.sgc import SGC
    from graphgallery.nn.models.semisupervised.TF.gat import GAT
    from graphgallery.nn.models.semisupervised.TF.clustergcn import ClusterGCN
    from graphgallery.nn.models.semisupervised.TF.gwnn import GWNN
    from graphgallery.nn.models.semisupervised.TF.robustgcn import RobustGCN
    from graphgallery.nn.models.semisupervised.TF.graphsage import GraphSAGE
    from graphgallery.nn.models.semisupervised.TF.fastgcn import FastGCN
    from graphgallery.nn.models.semisupervised.TF.chebynet import ChebyNet
    from graphgallery.nn.models.semisupervised.TF.densegcn import DenseGCN
    from graphgallery.nn.models.semisupervised.TF.lgcn import LGCN
    from graphgallery.nn.models.semisupervised.TF.obvat import OBVAT
    from graphgallery.nn.models.semisupervised.TF.sbvat import SBVAT
    from graphgallery.nn.models.semisupervised.TF.gmnn import GMNN
    from graphgallery.nn.models.semisupervised.TF.dagnn import DAGNN

    # experimental model
    from graphgallery.nn.models.semisupervised.experimental.mediansage import MedianSAGE
    from graphgallery.nn.models.semisupervised.experimental.edgeconv import EdgeGCN
    from graphgallery.nn.models.semisupervised.experimental.s_obvat import SimplifiedOBVAT
    from graphgallery.nn.models.semisupervised.experimental.gcn_mix import GCN_MIX
    from graphgallery.nn.models.semisupervised.experimental.gcna import GCNA
else:
    from graphgallery.nn.models.semisupervised.PTH.gcn import GCN

# unsupervised model
from graphgallery.nn.models.unsupervised.unsupervised_model import UnsupervisedModel
from graphgallery.nn.models.unsupervised.node2vec import Node2vec
from graphgallery.nn.models.unsupervised.deepwalk import Deepwalk
