from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery.nodeclas import Trainer
from graphgallery.nn.models import get_model


@PyTorch.register()
class GCN_VAT(Trainer):
    """
        Implementation of Graph Convolutional Networks (GCN) with Virtual Adversarial Training (VAT).
        `Graph Adversarial Training: Dynamically Regularizing Based on Graph Structure
        <https://arxiv.org/abs/1902.08226>`
        Tensorflow 1.x implementation: <https://github.com/fulifeng/GraphAT>
    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.data_device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=False,
                   xi=1e-4,                   
                   alpha=1.0,
                   epsilon=5e-2,
                   num_power_iterations=1):

        model = get_model("graphat.GCN_VAT", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      xi=xi,
                      alpha=alpha,
                      epsilon=epsilon,
                      num_power_iterations=num_power_iterations,
                      hids=hids,
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
                                     out_weight=index,
                                     device=self.data_device)
        return sequence