# Base model
from graphgallery.nn.models.basemodel import BaseModel

# (semi-)supervised model
from graphgallery.nn.models.supervised.supervised_model import SupervisedModel
from graphgallery.nn.models.supervised.gcn import GCN
from graphgallery.nn.models.supervised.gcn_mix import GCN_MIX
from graphgallery.nn.models.supervised.sgc import SGC
from graphgallery.nn.models.supervised.gat import GAT
from graphgallery.nn.models.supervised.clustergcn import ClusterGCN
from graphgallery.nn.models.supervised.gwnn import GWNN
from graphgallery.nn.models.supervised.robustgcn import RobustGCN
from graphgallery.nn.models.supervised.graphsage import GraphSAGE
from graphgallery.nn.models.supervised.fastgcn import FastGCN
from graphgallery.nn.models.supervised.chebynet import ChebyNet
from graphgallery.nn.models.supervised.densegcn import DenseGCN
from graphgallery.nn.models.supervised.lgcn import LGCN
from graphgallery.nn.models.supervised.obvat import OBVAT
from graphgallery.nn.models.supervised.sbvat import SBVAT
from graphgallery.nn.models.supervised.s_obvat import SimplifiedOBVAT
from graphgallery.nn.models.supervised.gmnn import GMNN
from graphgallery.nn.models.supervised.edgeconv import EdgeGCN
from graphgallery.nn.models.supervised.mediansage import MedianSAGE
from graphgallery.nn.models.supervised.gcnf import GCNF
from graphgallery.nn.models.supervised.robustgcnf import RobustGCNF


# unsupervised model
from graphgallery.nn.models.unsupervised.unsupervised_model import UnsupervisedModel
from graphgallery.nn.models.unsupervised.deepwalk import Deepwalk
from graphgallery.nn.models.unsupervised.node2vec import Node2vec
