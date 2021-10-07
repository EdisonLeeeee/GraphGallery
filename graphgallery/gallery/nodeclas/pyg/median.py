from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyG
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyG.register()
class MedianGCN(Trainer):
    """
        Implementation of Graph Convolutional Networks with Median aggregation (MedianGCN). 
        `Understanding Structural Vulnerability in Graph Convolutional Networks 
        <https://arxiv.org/abs/2108.06280>`
        Pytorch implementation: <https://github.com/EdisonLeeeee/MedianGCN>

    """

    def data_step(self,
                  adj_transform="add_selfloops",
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, E = gf.astensors(node_attr, adj_matrix, device=self.data_device)

        # ``E`` and ``X`` are cached for later use
        self.register_cache(X=X, E=E)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=True):

        model = get_model("MedianGCN", self.backend)
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
        sequence = FullBatchSequence([self.cache.X, *self.cache.E],
                                     labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
