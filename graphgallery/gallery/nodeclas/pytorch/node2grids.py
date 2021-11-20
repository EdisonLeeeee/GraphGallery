import torch
import graphgallery.nn.models.pytorch as models
from graphgallery.data.sequence import FeatureLabelSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery.nodeclas import NodeClasTrainer
from graphgallery.functional.graph_level import Node2GridsMapper


@PyTorch.register()
class Node2Grids(NodeClasTrainer):
    """
        Implementation of Node2Gridss.
        `Node2Grids: A Cost-Efficient Uncoupled Training Framework for Large-Scale Graph Learning`
        `An Uncoupled Training Architecture for Large Graph Learning <https://arxiv.org/abs/2003.09638>`
        Pytorch implementation: <https://github.com/Ray-inthebox/Node2Gridss>

    """

    def data_step(self,
                  adj_transform=None,
                  feat_transform=None,
                  biasfactor=0.4,
                  mapsize_a=12,
                  mapsize_b=1):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)
        mapper = Node2GridsMapper(adj_matrix, attr_matrix, biasfactor=biasfactor,
                                  mapsize_a=mapsize_a, mapsize_b=mapsize_b)
        self.register_cache(mapper=mapper, mapsize_a=mapsize_a, mapsize_b=mapsize_b)

    def model_step(self,
                   hids=[200],
                   acts=['relu6'],
                   dropout=0.6,
                   attnum=10,
                   conv_channel=64,
                   bias=True):

        cache = self.cache
        model = models.Node2GridsCNN(self.graph.num_feats,
                                     self.graph.num_classes,
                                     cache.mapsize_a, cache.mapsize_b,
                                     hids=hids,
                                     acts=acts,
                                     conv_channel=conv_channel,
                                     dropout=dropout,
                                     attnum=attnum,
                                     bias=bias)
        return model

    def train_step(self, dataloader) -> dict:
        """One-step training on the input dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            the trianing dataloader

        Returns
        -------
        dict
            the output logs, including `loss` and `val_accuracy`, etc.
        """
        loss_fn = self.loss
        model = self.model

        self.reset_metrics()
        model.train()

        att_reg = self.cfg.get("att_reg", 0.07)

        for epoch, batch in enumerate(dataloader):
            self.callbacks.on_train_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)

            if not isinstance(x, tuple):
                x = x,
            out = model(*x)
            if out_index is not None:
                out = out[out_index]
            loss = loss_fn(out, y) + att_reg * torch.sum(model.attention.view(-1).square())
            # here I exactly follow the author's implementation in
            # <https://github.com/Ray-inthebox/Node2Gridss>
            # But what is it????
            loss.backward(loss)
            for metric in self.metrics:
                metric.update_state(y.cpu(), out.detach().cpu())
            self.callbacks.on_train_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))

    def config_train_data(self, index):
        batch_size_train = self.cfg.get("batch_size_train", 8)
        labels = self.graph.label[index]
        node_grids = self.cache.mapper.map_node(index).transpose((0, 3, 1, 2))
        sequence = FeatureLabelSequence(node_grids, labels, device=self.data_device, batch_size=batch_size_train, shuffle=False)
        return sequence

    def config_test_data(self, index):
        batch_size_test = self.cfg.get("batch_size_test", 1000)
        labels = self.graph.label[index]
        node_grids = self.cache.mapper.map_node(index).transpose((0, 3, 1, 2))
        sequence = FeatureLabelSequence(node_grids, labels, device=self.data_device, batch_size=batch_size_test)
        return sequence

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.008)
        weight_decay = self.cfg.get('weight_decay', 0.00015)
        model = self.model
        return torch.optim.RMSprop(model.parameters(),
                                   weight_decay=weight_decay, lr=lr)
