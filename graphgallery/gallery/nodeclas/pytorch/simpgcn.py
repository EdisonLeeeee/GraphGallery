import torch
import numpy as np
import torch.nn.functional as F
import graphgallery.nn.models.pytorch as models

from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer


@PyTorch.register()
class SimPGCN(Trainer):
    """
        Implementation of Similarity Preserving Graph Convolutional Networks (SimPGCN).
        `Node Similarity Preserving Graph Convolutional Networks
        <https://arxiv.org/abs/2011.09643>`
        Pytorch implementation: <https://github.com/ChandlerBang/SimP-GCN>
    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  feat_transform=None,
                  recalculate=True):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)

        feat, adj = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``adj`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, adj=adj)

        if recalculate:
            # Uses this to save time for structure evation attack
            # NOTE: Please make sure the node attribute matrix remains the same if recalculate=False
            knn_graph = gf.normalize_adj(gf.knn_graph(attr_matrix), add_self_loop=False)
            pseudo_labels, node_pairs = gf.attr_sim(attr_matrix)
            knn_graph, pseudo_labels = gf.astensors(knn_graph, pseudo_labels, device=self.data_device)

            self.register_cache(knn_graph=knn_graph, pseudo_labels=pseudo_labels, node_pairs=node_pairs)

    def model_step(self,
                   hids=[64],
                   acts=[None],
                   dropout=0.5,
                   gamma=0.1,
                   bias=False):

        model = models.SimPGCN(self.graph.num_feats,
                               self.graph.num_classes,
                               hids=hids,
                               acts=acts,
                               gamma=gamma,
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
        optimizer = self.optimizer
        loss_fn = self.loss
        model = self.model

        self.reset_metrics()
        model.train()

        lambda_ = self.cfg.get("lambda_", 5.0)

        for epoch, batch in enumerate(dataloader):
            self.callbacks.on_train_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)
            optimizer.zero_grad()

            if not isinstance(x, tuple):
                x = x,
            out, embeddings = model(*x)
            if out_index is not None:
                out = out[out_index]

            y, pseudo_labels, node_pairs = y
            loss = loss_fn(out, y) + lambda_ * regression_loss(model, embeddings, pseudo_labels, node_pairs)
            loss.backward()
            optimizer.step()
            for metric in self.metrics:
                metric.update_state(y.cpu(), out.detach().cpu())
            self.callbacks.on_train_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))

    def config_train_data(self, index):

        labels = self.graph.label[index]
        cache = self.cache
        sequence = FullBatchSequence(inputs=[cache.feat, cache.adj, cache.knn_graph],
                                     y=[labels, cache.pseudo_labels, cache.node_pairs],
                                     out_index=index,
                                     device=self.data_device)
        return sequence

    def config_test_data(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.feat, self.cache.adj],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence


def regression_loss(model, embeddings, pseudo_labels, node_pairs):
    k = 10000
    if len(node_pairs[0]) > k:
        sampled = np.random.choice(len(node_pairs[0]), k, replace=False)

        embeddings0 = embeddings[node_pairs[0][sampled]]
        embeddings1 = embeddings[node_pairs[1][sampled]]
        embeddings = model.linear(torch.abs(embeddings0 - embeddings1))
        loss = F.mse_loss(embeddings, pseudo_labels[sampled].unsqueeze(-1), reduction='mean')
    else:
        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]
        embeddings = model.linear(torch.abs(embeddings0 - embeddings1))
        loss = F.mse_loss(embeddings, pseudo_labels.unsqueeze(-1), reduction='mean')
    return loss
