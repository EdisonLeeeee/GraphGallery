import numpy as np
import graphgallery.nn.models.pyg as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.linkpred import PyG
from graphgallery.gallery.linkpred import LinkpredTrainer


@PyG.register()
class GAE(LinkpredTrainer):
    """Implementation of Graph AutoEncoder (GAE) in
    `Variational Graph Auto-Encoders
    <https://arxiv.org/abs/1611.07308>`
    TensorFlow 1.x implementation <https://github.com/tkipf/gae>
    """

    def data_step(self, adj_transform=None, feat_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)

        feat = gf.astensor(attr_matrix, device=self.data_device)

        # ``feat`` is cached for later use
        self.register_cache(feat=feat, adj=adj_matrix)

    def model_step(self,
                   out_features=16,
                   hids=[32],
                   acts=['relu'],
                   dropout=0.,
                   bias=False):

        model = models.GAE(self.graph.num_feats,
                           out_features=out_features,
                           hids=hids,
                           acts=acts,
                           dropout=dropout,
                           bias=bias)

        return model

    def config_train_data(self, edge_index):
        if isinstance(edge_index, (list, tuple)):
            train_edges = edge_index[0]  # postive edge index
        else:
            train_edges = edge_index

        train_edges = gf.astensor(train_edges, device=self.data_device)

        self.register_cache(edges=train_edges)
        sequence = FullBatchSequence([self.cache.feat, train_edges],
                                     out_index=edge_index,
                                     device=self.data_device)
        return sequence

    def config_test_data(self, edge_index):

        if isinstance(edge_index, (list, tuple)):
            edge_index = np.hstack(edge_index)

        y = self.cache.adj[edge_index[0], edge_index[1]].A1
        y[y > 0] = 1

        sequence = FullBatchSequence([self.cache.feat, self.cache.edges],
                                     y=y,
                                     out_index=edge_index,
                                     device=self.data_device)
        return sequence

    def config_predict_data(self, edge_index):
        if isinstance(edge_index, (list, tuple)):
            edge_index = np.hstack(edge_index)

        sequence = FullBatchSequence([self.cache.feat, self.cache.edges],
                                     out_index=edge_index,
                                     device=self.data_device)
        return sequence


@PyG.register()
class VGAE(GAE):
    """Implementation of Variational Graph AutoEncoder (VGAE) in
    `Variational Graph Auto-Encoders
    <https://arxiv.org/abs/1611.07308>`
    TensorFlow 1.x implementation <https://github.com/tkipf/gae>
    """

    def model_step(self,
                   out_features=16,
                   hids=[32],
                   acts=['relu'],
                   dropout=0.,
                   bias=False):

        model = models.VGAE(self.graph.num_feats,
                            out_features=out_features,
                            hids=hids,
                            acts=acts,
                            dropout=dropout,
                            bias=bias)

        return model

    def train_step(self, dataloader) -> dict:
        loss_fn = self.loss
        model = self.model

        self.reset_metrics()
        model.train()

        for epoch, batch in enumerate(dataloader):
            self.callbacks.on_train_batch_begin(epoch)
            x, _, out_index = self.unravel_batch(batch)
            x = self.to_device(x)

            if not isinstance(x, tuple):
                x = x,

            z, mu, logstd = model(*x)

            # here `out_index` maybe pos_edge_index
            # or (pos_edge_index, neg_edge_index)
            if isinstance(out_index, (list, tuple)):
                assert len(out_index) == 2, '`out_index` should be (pos_edge_index, neg_edge_index) or pos_edge_index'
                pos_edge_index, neg_edge_index = out_index
            else:
                pos_edge_index, neg_edge_index = out_index, None

            pos_pred = model.decoder(z, pos_edge_index)
            pos_y = z.new_ones(pos_edge_index.size(1))

            if neg_edge_index is None:
                neg_edge_index = negative_sampling(pos_edge_index, z.size(0))

            neg_pred = model.decoder(z, neg_edge_index)
            neg_y = z.new_zeros(neg_edge_index.size(1))
            out = torch.cat([pos_pred, neg_pred], dim=0)
            y = torch.cat([pos_y, neg_y], dim=0)
            kl_loss = -0.5 / mu.size(0) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), dim=1))
            loss = loss_fn(out, y) + kl_loss
            loss.backward()
            for metric in self.metrics:
                metric.update_state(y.cpu(), out.detach().cpu())
            self.callbacks.on_train_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))
