from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model

from graphgallery.gallery.nodeclas import DGL


@DGL.register()
class DAGNN(Trainer):
    """
        Implementation of Deep Adaptive Graph Neural Network (DAGNN). 
        `Towards Deeper Graph Neural Networks <https://arxiv.org/abs/2007.09296>`
        Pytorch implementation: <https://github.com/mengliu1998/DeeperGNN>
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
                   hids=[64],
                   acts=['relu'],
                   dropout=0.5,
                   weight_decay=5e-3,
                   lr=0.01,
                   bias=False,
                   K=10):

        model = get_model("DAGNN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias,
                      K=K)

        return model

    def train_loader(self, index):
        labels = self.graph.node_label[index]
        sequence = FullBatchSequence(inputs=[self.cache.X, self.cache.G],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device,
                                     escape=type(self.cache.G))
        return sequence
