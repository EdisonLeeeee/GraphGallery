import graphgallery.nn.models.pytorch as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer


@PyTorch.register()
class AGNN(Trainer):
    """
        Implementation of Attention-based Graph Neural Network (AGNN).
        ` Attention-based Graph Neural Network for semi-supervised learning
        <https://arxiv.org/abs/1609.02907>`
        Pytorch implementation: <https://github.com/dawnranger/pytorch-AGNN>
    """

    def data_step(self,
                  adj_transform="add_self_loop",
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(attr_transform)(graph.attr_matrix)

        feat, adj = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``adj`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, adj=adj)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   num_attn=3,
                   dropout=0.5,
                   bias=False):

        model = models.AGNN(self.graph.num_feats,
                            self.graph.num_classes,
                            hids=hids,
                            acts=acts,
                            num_attn=num_attn,
                            dropout=dropout,
                            bias=bias)

        return model

    def config_train_data(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.feat, self.cache.adj],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
