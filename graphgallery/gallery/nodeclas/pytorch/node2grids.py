from graphgallery.nn.layers.pytorch import SGConv
from graphgallery.sequence import FeatureLabelSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model
from graphgallery.functional.graph_level import Node2GridsMapper


@PyTorch.register()
class Node2Grids(Trainer):
    """
        Implementation of Node2Gridss. 
        `Node2Gridss: A Cost-Efficient Uncoupled Training Framework for Large-Scale Graph Learning`
        `An Uncoupled Training Architecture for Large Graph Learning <https://arxiv.org/abs/2003.09638>`
        Pytorch implementation: <https://github.com/Ray-inthebox/Node2Gridss>

    """

    def custom_setup(self,
                     batch_size_train=100,
                     batch_size_test=1000):

        self.cfg.fit.batch_size = batch_size_train
        self.cfg.evaluate.batch_size = batch_size_test

    def data_step(self,
                  adj_transform=None,
                  attr_transform=None,
                  biasfactor=0.4,
                  mapsize_a=12,
                  mapsize_b=1):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)
        mapper = Node2GridsMapper(adj_matrix, node_attr, biasfactor=biasfactor,
                                  mapsize_a=mapsize_a, mapsize_b=mapsize_b)

        self.register_cache(mapper=mapper, mapsize_a=mapsize_a, mapsize_b=mapsize_b)

    def model_step(self,
                   hids=[200],
                   acts=['relu6'],
                   dropout=0.5,
                   attnum=10,
                   weight_decay=0.00015,
                   att_reg=0.07,
                   lr=0.008,
                   bias=True):

        cache = self.cache
        model = get_model("Node2GridsCNN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      cache.mapsize_a, cache.mapsize_b,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      att_reg=att_reg,
                      attnum=attnum,
                      bias=bias,
                      weight_decay=weight_decay,
                      lr=lr)
        return model

    def train_loader(self, index):
        labels = self.graph.node_label[index]
        node_grids = self.cache.mapper.map_node(index).transpose((0, 3, 1, 2))
        sequence = FeatureLabelSequence(node_grids, labels, device=self.data_device, batch_size=self.cfg.fit.batch_size, shuffle=False)
        return sequence

    def test_loader(self, index):
        labels = self.graph.node_label[index]
        node_grids = self.cache.mapper.map_node(index).transpose((0, 3, 1, 2))
        sequence = FeatureLabelSequence(node_grids, labels, device=self.data_device, batch_size=self.cfg.evaluate.batch_size)
        return sequence
