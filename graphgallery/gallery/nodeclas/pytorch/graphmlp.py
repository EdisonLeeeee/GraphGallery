import numpy as np
import torch
import graphgallery.nn.models.pytorch as models
from graphgallery.data.sequence import FullBatchSequence, Sequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer


@PyTorch.register()
class GraphMLP(Trainer):
    """
        Implementation of Graph-MLP.
        `Graph-MLP: Node Classification without Message Passing in Graph
        <https://arxiv.org/abs/2106.04051>`
        Pytorch implementation: <https://github.com/yanghu819/Graph-MLP>

    """

    def data_step(self,
                  adj_transform=("normalize_adj",
                                 ("adj_power", dict(power=2)),
                                 "to_dense"),
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(attr_transform)(graph.attr_matrix)

        feat, adj = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``adj`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, adj=adj)

    def model_step(self,
                   hids=[256],
                   acts=['gelu'],
                   dropout=0.6,
                   bias=True):

        model = models.GraphMLP(self.graph.num_feats,
                                self.graph.num_classes,
                                hids=hids,
                                acts=acts,
                                dropout=dropout,
                                bias=bias)

        return model

    def config_train_data(self, index):
        block_size = self.cfg.get('block_size', 2000)

        labels = self.graph.label[index]
        sequence = DenseBatchSequence(inputs=[self.cache.feat, self.cache.adj],
                                      y=labels,
                                      out_index=index,
                                      block_size=block_size,
                                      device=self.data_device)
        return sequence

    def train_step(self, dataloader) -> dict:
        optimizer = self.optimizer
        loss_fn = self.loss
        model = self.model

        optimizer.zero_grad()
        self.reset_metrics()
        model.train()

        alpha = self.cfg.get('alpha', 10.0)
        tau = self.cfg.get('tau', 2)

        for epoch, batch in enumerate(dataloader):
            self.callbacks.on_train_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)
            if not isinstance(x, tuple):
                x = x,

            # ====== additional contrast loss ===============
            out, h = model(*x)

            if out_index is not None:
                out = out[out_index]
            x_dis = get_feature_dis(h)
            loss = loss_fn(out, y[0]) + Ncontrast(x_dis, y[1], tau=tau) * alpha
            # ===============================================

            loss.backward()
            optimizer.step()
            for metric in self.metrics:
                metric.update_state(y[0].cpu(), out.detach().cpu())
            self.callbacks.on_train_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))

    def config_test_data(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=self.cache.feat,
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.001)
        weight_decay = self.cfg.get('weight_decay', 5e-3)
        model = self.model
        return torch.optim.Adam(model.parameters(),
                                weight_decay=weight_decay, lr=lr)


def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x @ x.T
    mask = torch.eye(x_dis.size(0), device=x.device)
    x_sum = torch.sum(x**2, 1).view(-1, 1)
    x_sum = torch.sqrt(x_sum).view(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum**(-1))
    x_dis = (1 - mask) * x_dis
    return x_dis


def Ncontrast(x_dis, adj_label, tau=1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))).mean()
    return loss


class DenseBatchSequence(Sequence):

    def __init__(self, inputs, y=None, out_index=None, block_size=2000, device='cpu', escape=None, **kwargs):
        dataset = gf.astensors(inputs, y, out_index, device=device, escape=escape)
        super().__init__([dataset], batch_size=None, collate_fn=self.collate_fn, device=device, escape=escape, **kwargs)
        self.block_size = block_size

    def collate_fn(self, dataset):
        (x, adj), y, out_index = dataset
        rand_indx = self.astensor(np.random.choice(np.arange(adj.shape[0]), self.block_size))
        # FIXME: there would be some error when out_index_i > len(rand_indx)
        if out_index is not None:
            rand_indx[:out_index.size(0)] = out_index
        features_batch = x[rand_indx]
        adj_label_batch = adj[rand_indx, :][:, rand_indx]
        return features_batch, (y, adj_label_batch), slice(0, out_index.size(0))
