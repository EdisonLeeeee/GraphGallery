from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model

from graphgallery.gallery.nodeclas import DGL


@DGL.register()
class APPNP(Trainer):
    """Implementation of approximated personalized propagation of neural 
        predictions (APPNP).
        `Predict then Propagate: Graph Neural Networks meet Personalized
        PageRank" <https://arxiv.org/abs/1810.05997>`
        Tensorflow 1.x implementation: <https://github.com/klicperajo/ppnp>
        Pytorch implementation: <https://github.com/klicperajo/ppnp>
    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  feat_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)
        X, G = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``G`` and ``X`` are cached for later use
        self.register_cache(X=X, G=G)

    def model_step(self,
                   hids=[64],
                   acts=['relu'],
                   alpha=0.1,
                   K=10,
                   ppr_dropout=0.,
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=True):

        model = get_model("APPNP", self.backend)
        model = model(self.graph.num_feats,
                      self.graph.num_classes,
                      hids=hids,
                      acts=acts,
                      alpha=alpha,
                      K=K,
                      ppr_dropout=ppr_dropout,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
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
