from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyTorch.register()
class TAGCN(Trainer):
    """
        Implementation of Topology Adaptive Graph Convolutional Networks <https://arxiv.org/abs/1710.10370> 
        Tensorflow 1.x implementation: <https://github.com/krohak/TAGCN>
        Create a Topology Adaptive Graph Convolutional Networks
         (TAGCN) model.
    """

    def data_step(self,
                  adj_transform=("normalize_adj",
                                 dict(fill_weight=0.0)),
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.data_device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def model_step(self,
                   hids=[16],
                   K=3,
                   acts=['relu'],
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=True):

        model = get_model("TAGCN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      K=K,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence(x=[self.cache.X, self.cache.A],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
