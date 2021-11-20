import torch
import torch.nn as nn
import graphgallery.nn.models.pytorch as models
from graphgallery.data.sequence import NullSequence, FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery.nodeclas import NodeClasTrainer
from .utils import *


@PyTorch.register()
class GraphVAT(NodeClasTrainer):
    """
        Implementation of Graph Convolutional Networks (GCN) with Virtual Adversarial Training (VAT).
        `Graph Adversarial Training: Dynamically Regularizing Based on Graph Structure
        <https://arxiv.org/abs/1902.08226>`
        Tensorflow 1.x implementation: <https://github.com/fulifeng/GraphAT>
    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  feat_transform=None,
                  num_neighbors=2):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)

        feat, adj = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)
        self.sampler = gf.ToNeighborMatrix(num_neighbors, self_loop=False, add_dummy=False)

        # ``adj`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, adj=adj, adjacency=adj_matrix)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.,
                   bias=False):

        model = models.GCN(self.graph.num_feats,
                           self.graph.num_classes,
                           hids=hids,
                           acts=acts,
                           dropout=dropout,
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

        alpha = self.cfg.get("alpha", 0.5)
        beta = self.cfg.get("beta", 1.0)

        for epoch, batch in enumerate(dataloader):
            self.callbacks.on_train_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)

            feat, adj, adjacency = x
            # this is used to calculate the adversarial gradients
            feat = nn.Parameter(feat)

            out = z = model(feat, adj)
            if out_index is not None:
                out = out[out_index]

            neighbors = torch.LongTensor(self.sampler(adjacency))

            loss = loss_fn(out, y)
            loss += alpha * self.virtual_adversarial_loss((feat, adj), z) + \
                beta * self.graph_adversarial_loss((feat, adj), z, neighbors)

            loss.backward()
            for metric in self.metrics:
                metric.update_state(y.cpu(), out.detach().cpu())
            self.callbacks.on_train_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))

    def generate_virtual_adversarial_perturbation(self, inputs, logit):
        x, adj = inputs
        d = nn.Parameter(torch.randn_like(x))
        xi = self.cfg.get("xi", 1e-5)
        epsilon = self.cfg.get("epsilon", 5e-2)
        num_power_iterations = self.cfg.get("num_power_iterations", 1)
        for _ in range(num_power_iterations):
            d = xi * l2_normalize(d)
            logit_p = logit
            logit_m = self.model(x + d, adj)
            dist = kld_with_logits(logit_p, logit_m)
            d = torch.autograd.grad(dist, d)[0].detach()

        return epsilon * l2_normalize(d)

    def virtual_adversarial_loss(self, inputs, logit):
        feat, adj = inputs
        r_adv = self.generate_virtual_adversarial_perturbation(inputs, logit)
        logit_p = logit.detach()
        logit_q = self.model(feat + r_adv, adj)
        return kld_with_logits(logit_p, logit_q)

    def generate_graph_adversarial_perturbation(self, feat, logit, neighbor_logits):
        dist = neighbor_kld_with_logit(neighbor_logits, logit)
        d = torch.autograd.grad(dist, feat, retain_graph=True)[0].detach()
        epsilon_graph = self.cfg.get("epsilon_graph", 1e-2)
        return epsilon_graph * l2_normalize(d)

    def graph_adversarial_loss(self, inputs, logit, neighbors):
        feat, adj = inputs
        neighbor_logits = logit[neighbors.t()].detach()
        r_gadv = self.generate_graph_adversarial_perturbation(feat, logit, neighbor_logits)
        logit_m = self.model(feat + r_gadv, adj)
        gat_loss = neighbor_kld_with_logit(neighbor_logits, logit_m)
        return gat_loss

    def config_train_data(self, index):

        labels = self.graph.label[index]
        sequence = NullSequence([self.cache.feat, self.cache.adj, self.cache.adjacency],
                                gf.astensor(labels, device=self.data_device),
                                gf.astensor(index, device=self.data_device),
                                device=self.data_device)
        return sequence

    def config_test_data(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.feat, self.cache.adj],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
