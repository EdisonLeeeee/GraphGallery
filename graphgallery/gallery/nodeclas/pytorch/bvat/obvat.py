import torch
import torch.nn.functional as F
import graphgallery.nn.models.pytorch as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery.nodeclas import NodeClasTrainer
from ..utils import *


@PyTorch.register()
class OBVAT(NodeClasTrainer):
    """
        Implementation of optimization-based Batch Virtual Adversarial Training 
        Graph Convolutional Networks (OBVAT).
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
        self.register_cache(feat=feat, adj=adj)

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

        model.r_adv = torch.nn.Parameter(torch.zeros(
            self.graph.num_nodes, self.graph.num_feats))

        return model

    def config_train_data(self, index):
        labels = self.graph.label[index]
        sequence = FullBatchSequence([self.cache.feat, self.cache.adj],
                                     labels,
                                     out_index=index,
                                     device=self.data_device)

        return sequence

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.01)
        weight_decay = self.cfg.get('weight_decay', 5e-4)
        model = self.model
        self.adv_optimizer = torch.optim.Adam([model.r_adv], lr=lr / 10)
        return torch.optim.Adam([dict(params=model.reg_paras,
                                      weight_decay=weight_decay),
                                 dict(params=model.non_reg_paras,
                                      weight_decay=0.)], lr=lr)

    def pretrain(self, x):
        model = self.model
        self.freeze(model.conv)

        optimizer = self.adv_optimizer
        r_adv = model.r_adv
        for _ in range(10):
            optimizer.zero_grad()
            z = model(*x)
            rnorm = r_adv.square().sum()  # l2 loss
            loss = rnorm - self.virtual_adversarial_loss(x, z)
            r_adv.grad = torch.autograd.grad(loss, r_adv)[0]
            optimizer.step()

        self.defrozen(model.conv)

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

        p1 = self.cfg.get("p1", 1.4)
        p2 = self.cfg.get("p2", 0.7)

        for epoch, batch in enumerate(dataloader):

            self.callbacks.on_train_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)

            self.pretrain(x)

            if not isinstance(x, tuple):
                x = x,

            out = z = model(*x)
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

    def virtual_adversarial_loss(self, inputs, logit):
        feat, adj = inputs
        model = self.model
        logit_p = logit.detach()
        logit_q = model(feat + model.r_adv, adj)
        return masked_kld_with_logits(logit_p, logit_q)

    def entropy_loss(self, logit):
        q = F.softmax(logit, dim=-1)
        cross_entropy = softmax_cross_entropy_with_logits(
            logits=logit, labels=q)
        return cross_entropy.mean()
