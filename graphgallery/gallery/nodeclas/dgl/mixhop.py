from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model

from graphgallery.gallery.nodeclas import DGL


@DGL.register()
class MixHop(Trainer):
    """
        Implementation of MixHop Graph Convolutional layer from paper `MixHop: Higher-Order Graph Convolutional Architecturesvia Sparsified Neighborhood Mixing
        <https://arxiv.org/abs/1905.00067>`
        Tensorflow  implementation: <https://github.com/samihaija/mixhop>
    """

    def data_step(self,
                  adj_transform="add_self_loop",
                  attr_transform=None):
        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(attr_transform)(graph.attr_matrix)
        X, G = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``G`` and ``X`` are cached for later use
        self.register_cache(X=X, G=G)

    def model_step(self, hids=[60] * 3,
                   acts=['tanh'] * 3,
                   p=[0, 1, 2],
                   dropout=0.5,
                   weight_decay=5e-4,
                   bias=False,
                   lr=0.1,
                   step_size=40,
                   gamma=0.01):

        model = get_model("MixHop", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      p=p,
                      step_size=step_size,
                      gamma=gamma,
                      bias=bias)

        return model

    def train_loader(self, index):
        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.X, self.cache.G],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device,
                                     escape=type(self.cache.G))
        return sequence
