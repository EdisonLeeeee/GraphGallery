# Base model
from graphgallery.nn.models.base import Base
from graphgallery.nn.models.base_model import BaseModel
from graphgallery.nn.models.torch_keras_model import TorchKerasModel

# (semi-)supervised model
from graphgallery.nn.models.semisupervised.semi_supervised_model import SemiSupervisedModel

from graphgallery.nn.models.semisupervised.gcn import GCN
from graphgallery.nn.models.semisupervised.gat import GAT
from graphgallery.nn.models.semisupervised.clustergcn import ClusterGCN


### only tensorflow models#######################################################
from graphgallery.nn.models.semisupervised.sgc import SGC
from graphgallery.nn.models.semisupervised.gwnn import GWNN
from graphgallery.nn.models.semisupervised.robustgcn import RobustGCN
from graphgallery.nn.models.semisupervised.graphsage import GraphSAGE
from graphgallery.nn.models.semisupervised.fastgcn import FastGCN
from graphgallery.nn.models.semisupervised.chebynet import ChebyNet
from graphgallery.nn.models.semisupervised.densegcn import DenseGCN
from graphgallery.nn.models.semisupervised.lgcn import LGCN
from graphgallery.nn.models.semisupervised.obvat import OBVAT
from graphgallery.nn.models.semisupervised.sbvat import SBVAT
from graphgallery.nn.models.semisupervised.gmnn import GMNN
from graphgallery.nn.models.semisupervised.dagnn import DAGNN

# experimental model
from graphgallery.nn.models.semisupervised.experimental.edgeconv import EdgeGCN
from graphgallery.nn.models.semisupervised.experimental.s_obvat import SimplifiedOBVAT
from graphgallery.nn.models.semisupervised.experimental.gcn_mix import GCN_MIX
from graphgallery.nn.models.semisupervised.experimental.gcna import GCNA
######################################################################################

# unsupervised model
from graphgallery.nn.models.unsupervised.unsupervised_model import UnsupervisedModel
from graphgallery.nn.models.unsupervised.node2vec import Node2vec
from graphgallery.nn.models.unsupervised.deepwalk import Deepwalk
