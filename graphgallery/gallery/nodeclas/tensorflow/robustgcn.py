from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import TensorFlow
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@TensorFlow.register()
class RobustGCN(Trainer):
    """
        Implementation of Robust Graph Convolutional Networks (RobustGCN). 
        `Robust Graph Convolutional Networks Against Adversarial Attacks 
        <https://dl.acm.org/doi/10.1145/3292500.3330851>`
        Tensorflow 1.x implementation: <https://github.com/thumanlab/nrlweb/blob/master/static/assets/download/RGCN.zip>

    """

    def data_step(self,
                  adj_transform=("normalize_adj", dict(rate=[-0.5, -1.0])),
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.data_device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def model_step(self,
                   hids=[64],
                   acts=['relu'],
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   kl=5e-4,
                   gamma=1.,
                   bias=False):

        model = get_model("RobustGCN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      kl=kl,
                      gamma=gamma,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence([self.cache.X, *self.cache.A],
                                     labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
