from graphgallery.nn.layers.pytorch import SGConv
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyTorch.register()
class SGC(Trainer):
    """
        Implementation of Simplifying Graph Convolutional Networks (SGC). 
        `Simplifying Graph Convolutional Networks <https://arxiv.org/abs/1902.07153>`
        Pytorch implementation: <https://github.com/Tiiiger/SGC>

    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None,
                  K=2):
        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(attr_transform)(graph.attr_matrix)

        X, A = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        X = SGConv(K=K)(X, A)
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
        model = model(self.graph.num_feats,
                      self.graph.num_classes,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):
        labels = self.graph.label[index]
        X = self.cache.X[index]
        sequence = FullBatchSequence(X, labels, device=self.data_device)
        return sequence
