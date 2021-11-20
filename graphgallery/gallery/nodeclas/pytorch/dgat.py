import torch
import torch.nn as nn
import graphgallery.nn.models.pytorch as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery.nodeclas import NodeClasTrainer
from .utils import *


@PyTorch.register()
class DGAT(NodeClasTrainer):
    """
        Implementation of Graph Convolutional Networks with directional graph adversarial training (DGAT).
        `Robust graph convolutional networks with directional graph adversarial training
        <https://link.springer.com/article/10.1007/s10489-021-02272-y>`

    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  feat_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix).A  # as dense matrix
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)

        feat, adj = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``adj`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, adj=adj)

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

        alpha = self.cfg.get("alpha", 1.0)

        for epoch, batch in enumerate(dataloader):
            self.callbacks.on_train_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)

            out = z = model(*x)
            if out_index is not None:
                out = out[out_index]

            loss = loss_fn(out, y)
            loss += alpha * self.dgat_loss(x, z)

            loss.backward()
            for metric in self.metrics:
                metric.update_state(y.cpu(), out.detach().cpu())
            self.callbacks.on_train_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))

    def dgat_loss(self, inputs, logit):
        feat, adj = inputs

        epsilon = self.cfg.get("epsilon", 0.9)

        adj_para = nn.Parameter(adj)
        feat_new = self.generate_feat(feat, adj)

        logit_p = logit.detach()
        logit_m = self.model(feat_new, adj_para)
        dist = kld_with_logits(logit_p, logit_m)
        grad = torch.autograd.grad(dist, adj_para, retain_graph=True)[0].detach()

        adj_new = epsilon * get_normalized_vector(grad * adj)
        feat_new = self.generate_feat(feat, adj_new)

        logit_p = logit.detach()
        logit_m = self.model(feat_new, adj)
        loss = kld_with_logits(logit_p, logit_m)
        return loss

    def generate_feat(self, feat, adj):
        D = torch.diag(adj.sum(dim=1))
        L = D - adj
        feat_new = (torch.eye(L.size(0), device=L.device) - L) @ feat
        return feat_new

    def config_train_data(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.feat, self.cache.adj],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
