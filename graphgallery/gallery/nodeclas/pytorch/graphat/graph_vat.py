from graphgallery.sequence import NullSequence, FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyTorch.register()
class GraphVAT(Trainer):
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
        self.register_cache(X=X, A=A, adjacency=adj_matrix)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=False,
                   xi=1e-5,
                   alpha=0.5,
                   beta=1.0,
                   num_neighbors=2,
                   epsilon=0.05,
                   num_power_iterations=1,
                   epsilon_graph=1e-2):

        model = get_model("graphat.GraphVAT", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      xi=xi,
                      alpha=alpha,
                      beta=beta,
                      epsilon=epsilon,
                      num_power_iterations=num_power_iterations,
                      epsilon_graph=epsilon_graph,
                      num_neighbors=num_neighbors,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):

        labels = self.graph.node_label[index]
        sequence = NullSequence([self.cache.X, self.cache.A, self.cache.adjacency],
                                gf.astensor(labels, device=self.data_device),
                                gf.astensor(index, device=self.data_device),
                                device=self.data_device)
        return sequence

    def test_loader(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence(inputs=[self.cache.X, self.cache.A],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
