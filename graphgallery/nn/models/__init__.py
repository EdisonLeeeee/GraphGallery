# Base model

from graphgallery.nn.models.base_model import base_model
from graphgallery import backend

_BACKEND = backend()

# (semi-)supervised model
from graphgallery.nn.models.semisupervised.semi_supervised_model import SemiSupervisedModel
if _BACKEND .kind == "T":

    from graphgallery.nn.models.semisupervised.tf_models.gcn import GCN
    from graphgallery.nn.models.semisupervised.tf_models.sgc import SGC
    from graphgallery.nn.models.semisupervised.tf_models.gat import GAT
    from graphgallery.nn.models.semisupervised.tf_models.clustergcn import ClusterGCN
    from graphgallery.nn.models.semisupervised.tf_models.gwnn import GWNN
    from graphgallery.nn.models.semisupervised.tf_models.robustgcn import RobustGCN
    from graphgallery.nn.models.semisupervised.tf_models.graphsage import GraphSAGE
    from graphgallery.nn.models.semisupervised.tf_models.fastgcn import FastGCN
    from graphgallery.nn.models.semisupervised.tf_models.chebynet import ChebyNet
    from graphgallery.nn.models.semisupervised.tf_models.densegcn import DenseGCN
    from graphgallery.nn.models.semisupervised.tf_models.lgcn import LGCN
    from graphgallery.nn.models.semisupervised.tf_models.obvat import OBVAT
    from graphgallery.nn.models.semisupervised.tf_models.sbvat import SBVAT
    from graphgallery.nn.models.semisupervised.tf_models.gmnn import GMNN
    from graphgallery.nn.models.semisupervised.tf_models.dagnn import DAGNN
    # experimental
    from graphgallery.nn.models.semisupervised.experimental.mediansage import MedianSAGE
    from graphgallery.nn.models.semisupervised.experimental.edgeconv import EdgeGCN
    from graphgallery.nn.models.semisupervised.experimental.s_obvat import SimplifiedOBVAT
    from graphgallery.nn.models.semisupervised.experimental.gcn_mix import GCN_MIX
    from graphgallery.nn.models.semisupervised.experimental.gcnf import GCNF
else:
    GCN = None

# unsupervised model
from graphgallery.nn.models.unsupervised.unsupervised_model import UnsupervisedModel
from graphgallery.nn.models.unsupervised.node2vec import Node2vec
from graphgallery.nn.models.unsupervised.deepwalk import Deepwalk
