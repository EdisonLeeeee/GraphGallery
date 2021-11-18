from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model

from graphgallery.gallery.nodeclas import DGL


@DGL.register()
class GRAND(Trainer):
    """
        Implementation of GRAND. 
        `Graph Random Neural Network for Semi-Supervised Learning on Graphs  
        <https://arxiv.org/pdf/2005.11079.pdf>`
        Pytorch implementation: <https://github.com/THUDM/GRAND>
    """

    def data_step(self,
                  adj_transform="add_self_loop",
                  feat_transform=None):
        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)
        X, G = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``G`` and ``X`` are cached for later use
        self.register_cache(X=X, G=G)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   S=1,
                   K=4, temp=0.5, lam=1.,
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=False,
                   bn=False):

        model = get_model("GRAND", self.backend)
        model = model(self.graph.num_feats,
                      self.graph.num_classes,
                      hids=hids,
                      acts=acts,
                      S=S,
                      K=K,
                      temp=temp,
                      lam=lam,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias,
                      bn=bn)

        return model

    def train_loader(self, index):
        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.X, self.cache.G],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device,
                                     escape=type(self.cache.G))
        return sequence
