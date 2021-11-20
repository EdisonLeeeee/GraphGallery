import graphgallery.nn.models.pytorch as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery.nodeclas import NodeClasTrainer


@PyTorch.register()
class GWNN(NodeClasTrainer):
    """
        Implementation of Graph Wavelet Neural Networks (GWNN). 
        `Graph Wavelet Neural Network <https://arxiv.org/abs/1904.07785>`
        Tensorflow 1.x implementation: <https://github.com/Eilene/GWNN>
        Pytorch implementation: 
        <https://github.com/benedekrozemberczki/GraphWaveletNeuralNetwork>

    """

    def data_step(self,
                  adj_transform="wavelet_basis",
                  feat_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)

        feat, adj = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``adj`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, adj=adj)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.5,
                   bias=False):

        model = models.GWNN(self.graph.num_feats,
                            self.graph.num_classes,
                            self.graph.num_nodes,
                            hids=hids,
                            acts=acts,
                            dropout=dropout,
                            bias=bias)

        return model

    def config_train_data(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence([self.cache.feat, *self.cache.adj],
                                     labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
