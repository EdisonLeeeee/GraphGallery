import torch
import graphgallery.nn.models.pytorch as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer


@PyTorch.register()
class RobustGCN(Trainer):
    """
        Implementation of Robust Graph Convolutional Networks (RobustGCN). 
        `Robust Graph Convolutional Networks Against Adversarial Attacks 
        <https://dl.acm.org/doi/10.1145/3292500.3330851>`
        Tensorflow 1.x implementation: <https://github.com/thumanlab/nrlweb/blob/master/static/assets/download/RGCN.zip>

    """

    def data_step(self,
                  adj_transform=("normalize_adj", dict(rate=[-0.5, -1.0])),
                  feat_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)

        X, A = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def model_step(self,
                   hids=[64],
                   acts=['relu'],
                   dropout=0.5,
                   gamma=1.,
                   bias=False):

        model = models.RobustGCN(self.graph.num_feats,
                                 self.graph.num_classes,
                                 hids=hids,
                                 acts=acts,
                                 dropout=dropout,
                                 gamma=gamma,
                                 bias=bias)

        return model

    def config_train_data(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence([self.cache.X, *self.cache.A],
                                     labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence

    def train_step(self, dataloader):
        loss_fn = self.loss
        model = self.model

        self.reset_metrics()
        model.train()

        kl = self.cfg.get('kl', 5e-4)

        for epoch, batch in enumerate(dataloader):
            self.callbacks.on_train_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)

            if not isinstance(x, tuple):
                x = x,
            out = model(*x)[out_index]
            # ================= add KL loss here =============================
            mean, var = model.mean, model.var
            kl_loss = -0.5 * torch.sum(torch.mean(1 + torch.log(var + 1e-8) -
                                                  mean.pow(2) + var, dim=1))
            loss = loss_fn(out, y) + kl * kl_loss
            # ===============================================================
            loss.backward()
            for metric in self.metrics:
                metric.update_state(y.cpu(), out.detach().cpu())
            self.callbacks.on_train_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))
