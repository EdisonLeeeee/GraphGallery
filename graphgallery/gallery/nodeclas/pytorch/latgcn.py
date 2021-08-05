from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyTorch.register()
class LATGCN(Trainer):
    """
        Implementation of Latent Adversarial Training of Graph Convolutional Networks (LATGCN).
        `Latent Adversarial Training of Graph Convolutional Networks
        <https://graphreason.github.io/papers/35.pdf>`
        Tensorflow 1.x implementation: <https://github1s.com/cshjin/LATGCN/>

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
                   dropout=0.2,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=False,
                   gamma=0.01,
                   eta=0.1):

        model = get_model("LATGCN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      self.graph.num_nodes,
                      gamma=gamma,
                      eta=eta,
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
                                     out_index=index,
                                     device=self.data_device)
        return sequence
