import graphgallery.nn.models.pytorch as models
from graphgallery.data.sequence import SAGESequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer


@PyTorch.register()
class GraphSAGE(Trainer):
    """
        Implementation of SAmple and aggreGatE Graph Convolutional Networks (GraphSAGE). 
        `Inductive Representation Learning on Large Graphs <https://arxiv.org/abs/1706.02216>`
        Tensorflow 1.x implementation: <https://github.com/williamleif/GraphSAGE>
        Pytorch implementation: <https://github.com/williamleif/graphsage-simple/>
    """

    def data_step(self,
                  adj_transform=None,
                  feat_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)

        feat, adj = gf.astensors(attr_matrix, device=self.data_device), adj_matrix

        # ``adj`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, adj=adj)

    def model_step(self,
                   hids=[32],
                   acts=['relu'],
                   dropout=0.5,
                   bias=False,
                   sizes=[15, 5],
                   output_normalize=False,
                   aggregator='mean'):

        self.sizes = sizes

        model = models.GraphSAGE(self.graph.num_feats,
                                 self.graph.num_classes,
                                 hids=hids,
                                 acts=acts,
                                 dropout=dropout,
                                 bias=bias,
                                 aggregator=aggregator,
                                 output_normalize=output_normalize,
                                 sizes=sizes)
        return model

    def config_train_data(self, index):
        labels = self.graph.label[index]
        batch_size_train = self.cfg.get("batch_size_train", 512)

        sequence = SAGESequence(
            inputs=[self.cache.feat, self.cache.adj],
            nodes=index,
            y=labels,
            shuffle=True,
            batch_size=batch_size_train,
            sizes=self.sizes,
            device=self.data_device)
        return sequence

    def config_test_data(self, index):
        labels = self.graph.label[index]
        batch_size_test = self.cfg.get("batch_size_test", 20000)
        sequence = SAGESequence(
            inputs=[self.cache.feat, self.cache.adj],
            nodes=index,
            y=labels,
            batch_size=batch_size_test,
            sizes=self.sizes,
            device=self.data_device)
        return sequence
