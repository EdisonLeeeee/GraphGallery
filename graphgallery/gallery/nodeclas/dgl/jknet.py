from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model

from graphgallery.gallery.nodeclas import DGL


@DGL.register()
class JKNet(Trainer):
    """
        Implementation of Jumping Knowledge Networks (JKNet). 
        `Representation Learning on Graphs with Jumping Knowledge Networks
        <https://arxiv.org/abs/1806.03536>`

        DGL implementation: <https://github.com/mori97/JKNet-dgl>
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
                   hids=[16] * 5, acts=['relu'] * 5,
                   mode='cat',
                   dropout=0.5, weight_decay=5e-4,
                   lr=0.005, bias=True):

        model = get_model("JKNet", self.backend)
        model = model(self.graph.num_feats,
                      self.graph.num_classes,
                      hids=hids,
                      acts=acts,
                      mode=mode,
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
