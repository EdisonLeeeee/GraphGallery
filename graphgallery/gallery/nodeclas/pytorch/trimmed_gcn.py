from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyTorch.register()
class TrimmedGCN(Trainer):
    """
        Implementation of Graph Convolutional Networks with Trimmed mean aggregation (TrimmedGCN). 
        `Understanding Structural Vulnerability in Graph Convolutional Networks 
        <https://arxiv.org/abs/2108.06280>`
        Pytorch implementation: <https://github.com/EdisonLeeeee/MedianGCN>

    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix.tolil().rows, device=self.data_device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.5,
                   weight_decay=1e-4,
                   lr=0.01,
                   tperc=0.45,
                   bias=False):

        model = get_model("TrimmedGCN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      tperc=tperc,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)
        return model

    def train_loader(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence([self.cache.X, self.cache.A],
                                     labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
