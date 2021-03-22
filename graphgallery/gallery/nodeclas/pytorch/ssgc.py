from graphgallery.nn.layers.pytorch import SSGConv
from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery.nodeclas import Trainer
from graphgallery.nn.models import get_model


@PyTorch.register()
class SSGC(Trainer):
    """
        Implementation of Simple Spectral Graph Convolution (SSGC). 
        `Simple Spectral Graph Convolution <https://openreview.net/forum?id=CYO5T-YjWZV>`
        Pytorch implementation: https://github.com/allenhaozhu/SSGC      

    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None,
                  K=16, 
                  alpha=0.1):
        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.data_device)

        X = SSGConv(K=K, alpha=alpha)(X, A)
        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def model_step(self,
                   hids=[],
                   acts=[],
                   dropout=0.5,
                   weight_decay=5e-5,
                   lr=0.2,
                   bias=True):

        model = get_model("MLP", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):
        labels = self.graph.node_label[index]
        X = self.cache.X[index]
        sequence = FullBatchSequence(X, labels, device=self.data_device)
        return sequence
