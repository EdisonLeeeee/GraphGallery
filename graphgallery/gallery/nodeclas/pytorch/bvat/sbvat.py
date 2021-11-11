from graphgallery.data.sequence import FullBatchSequence, SBVATSampleSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyTorch.register()
class SBVAT(Trainer):
    """
        Implementation of sample-based Batch Virtual Adversarial Training
        Graph Convolutional Networks (SBVAT).
        `Batch Virtual Adversarial Training for Graph Convolutional Networks
        <https://arxiv.org/abs/1902.09192>`
        Tensorflow 1.x implementation: <https://github.com/thudzj/BVAT>
    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None,
                  sizes=50):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(attr_transform)(graph.attr_matrix)

        X, A = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A, neighbors=gf.find_4o_nbrs(adj_matrix))

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=False,
                   xi=1e-6,
                   p1=1.0,
                   p2=1.0,
                   epsilon=3e-2,
                   num_power_iterations=1):

        model = get_model("bvat.SBVAT", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      xi=xi,
                      p1=p1,
                      p2=p2,
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
        labels = self.graph.label[index]
        sequence = SBVATSampleSequence(inputs=[self.cache.X, self.cache.A],
                                       neighbors=self.cache.neighbors,
                                       y=labels,
                                       out_index=index,
                                       sizes=self.cfg.data.sizes,
                                       device=self.data_device)

        return sequence

    def test_loader(self, index):
        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.X, self.cache.A],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)

        return sequence
