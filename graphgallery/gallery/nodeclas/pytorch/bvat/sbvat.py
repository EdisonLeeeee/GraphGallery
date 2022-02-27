import torch
import torch.nn as nn
import graphgallery.nn.models.pytorch as models
from graphgallery.data.sequence import FullBatchSequence, SBVATSampleSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery.nodeclas import NodeClasTrainer
from ..utils import *


@PyTorch.register()
class SBVAT(NodeClasTrainer):
    """
        Implementation of sample-based Batch Virtual Adversarial Training
        Graph Convolutional Networks (SBVAT).
        `Batch Virtual Adversarial Training for Graph Convolutional Networks
        <https://arxiv.org/abs/1902.09192>`
        Tensorflow 1.x implementation: <https://github.com/thudzj/BVAT>
    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  feat_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)

        feat, adj = gf.astensors(
            attr_matrix, adj_matrix, device=self.data_device)

        # ``adj`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, adj=adj,
                            neighbors=gf.find_4o_nbrs(adj_matrix))

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.5,
                   bias=False):

        model = models.GCN(self.graph.num_feats,
                           self.graph.num_classes,
                           hids=hids,
                           acts=acts,
                           dropout=dropout,
                           bias=bias)

        return model

    def config_train_data(self, index):
        labels = self.graph.label[index]
        sizes = self.cfg.get("sizes", 50)
        sequence = SBVATSampleSequence(inputs=[self.cache.feat, self.cache.adj],
                                       neighbors=self.cache.neighbors,
                                       y=labels,
                                       out_index=index,
                                       sizes=sizes,
                                       device=self.data_device)

        return sequence

    def config_test_data(self, index):
        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.feat, self.cache.adj],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)

        return sequence

    def train_step(self, dataloader) -> dict:
        """One-step training on the input dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            the training dataloader

        Returns
        -------
        dict
            the output logs, including `loss` and `val_accuracy`, etc.
        """
        loss_fn = self.loss
        model = self.model

        self.reset_metrics()
        model.train()

        p1 = self.cfg.get("p1", 1.0)
        p2 = self.cfg.get("p2", 1.0)

        for epoch, batch in enumerate(dataloader):

            self.callbacks.on_train_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)

            if not isinstance(x, tuple):
                x = x,
            out = z = model(*x[:-1])

            if out_index is not None:
                out = out[out_index]

            loss = loss_fn(out, y) + p1 * self.virtual_adversarial_loss(x, z) + \
                p2 * self.entropy_loss(z)
            loss.backward()
            for metric in self.metrics:
                metric.update_state(y.cpu(), out.detach().cpu())
            self.callbacks.on_train_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))

    def generate_virtual_adversarial_perturbation(self, inputs, logit):
        feat, adj, adv_mask = inputs
        d = nn.Parameter(torch.randn_like(feat))
        num_power_iterations = self.cfg.get("num_power_iterations", 1)
        xi = self.cfg.get("xi", 1e-6)
        epsilon = self.cfg.get("epsilon", 3e-2)

        for _ in range(num_power_iterations):
            d = xi * l2_normalize(d)
            logit_p = logit
            logit_m = self.model(feat + d, adj)
            dist = masked_kld_with_logits(logit_p, logit_m, adv_mask)
            d = torch.autograd.grad(dist, d)[0].detach()
        return epsilon * l2_normalize(d)

    def virtual_adversarial_loss(self, inputs, logit):
        feat, adj, adv_mask = inputs
        r_adv = self.generate_virtual_adversarial_perturbation(inputs, logit)
        logit_p = logit.detach()
        logit_q = self.model(feat + r_adv, adj)
        return masked_kld_with_logits(logit_p, logit_q, adv_mask)

    def entropy_loss(self, logit):
        q = F.softmax(logit, dim=-1)
        cross_entropy = softmax_cross_entropy_with_logits(
            logits=logit, labels=q)
        return cross_entropy.mean()
