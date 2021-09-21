from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model

from graphgallery.gallery.nodeclas import DGL


@DGL.register()
class GAT(Trainer):
    """
        Implementation of Graph Attention Networks (GAT).
        `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`
        Tensorflow 1.x implementation: <https://github.com/PetarV-/GAT>
        Pytorch implementation: <https://github.com/Diego999/pyGAT>
        Keras implementation: <https://github.com/danielegrattarola/keras-gat>

    """

    def data_step(self,
                  adj_transform="add_selfloops",
                  attr_transform=None):
        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)
        X, G = gf.astensors(node_attr, adj_matrix, device=self.data_device)

        # ``G`` and ``X`` are cached for later use
        self.register_cache(X=X, G=G)

    def model_step(self,
                   hids=[8],
                   num_heads=[8],
                   acts=['elu'],
                   dropout=0.6,
                   weight_decay=5e-4,
                   lr=0.01,
                   include=["num_heads"]):

        model = get_model("GAT", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      num_heads=num_heads,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr)

        return model

    def train_loader(self, index):
        labels = self.graph.node_label[index]
        sequence = FullBatchSequence([self.cache.X, self.cache.G],
                                     labels,
                                     out_index=index,
                                     device=self.data_device,
                                     escape=type(self.cache.G))
        return sequence
