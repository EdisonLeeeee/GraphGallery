from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import Trainer
from graphgallery.nn.models import get_model

from graphgallery.gallery.nodeclas import DGL_PyTorch


@DGL_PyTorch.register()
class GAT(Trainer):
    """
        Implementation of Graph Attention Networks (GAT).
        `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`
        Tensorflow 1.x implementation: <https://github.com/PetarV-/GAT>
        Pytorch implementation: <https://github.com/Diego999/pyGAT>
        Keras implementation: <https://github.com/danielegrattarola/keras-gat>

    """

    def process_step(self,
                     adj_transform="add_selfloops",
                     attr_transform=None,
                     graph_transform=None):
        graph = gf.get(graph_transform)(self.graph)
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)
        X, G = gf.astensors(node_attr, adj_matrix, device=self.device)

        # ``G`` and ``X`` are cached for later use
        self.register_cache(X=X, G=G)

    def builder(self,
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

    def train_sequence(self, index):
        labels = self.graph.node_label[index]
        sequence = FullBatchSequence([self.cache.X, self.cache.G],
                                     labels,
                                     out_weight=index,
                                     device=self.device,
                                     escape=type(self.cache.G))
        return sequence
