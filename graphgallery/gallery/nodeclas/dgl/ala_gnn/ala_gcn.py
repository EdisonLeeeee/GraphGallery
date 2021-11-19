import torch
import graphgallery.nn.models.dgl as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import Trainer
from graphgallery.gallery.nodeclas import DGL


@DGL.register()
class ALaGCN(Trainer):
    """
        Implementation of ALaGCN in
        `When Do GNNs Work: Understanding and Improving Neighborhood Aggregation 
        <https://www.ijcai.org/Proceedings/2020/0181.pdf>`
        DGL implementation: <https://github.com/raspberryice/ala-gcn>

    """

    def data_step(self,
                  adj_transform="add_self_loop",
                  feat_transform=None):
        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)
        feat, g = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``g`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, g=g)

    def model_step(self,
                   hids=[16] * 5,
                   acts=[None] * 5,
                   dropout=0.5,
                   share_tau=True,
                   bias=False):

        model = models.ala_gnn.ALaGCN(self.graph.num_feats,
                                      self.graph.num_classes,
                                      self.graph.num_nodes,
                                      hids=hids,
                                      acts=acts,
                                      dropout=dropout,
                                      share_tau=share_tau,
                                      bias=bias)

        return model

    def config_train_data(self, index):
        labels = self.graph.label[index]

        # ==========================================================
        # initial weight_y is obtained by linear regression
        feat = self.cache.feat.to(self.device)
        labels = gf.astensor(labels, device=self.device)
        A = torch.mm(feat.t(), feat) + 1e-05 * torch.eye(feat.size(1), device=feat.device)
        labels_one_hot = feat.new_zeros(feat.size(0), self.graph.num_classes)
        labels_one_hot[torch.arange(labels.size(0)), labels] = 1
        self.model.init_weight_y = torch.mm(
            torch.mm(torch.cholesky_inverse(A), feat.t()), labels_one_hot
        )
        # ==========================================================

        sequence = FullBatchSequence([self.cache.feat, self.cache.g],
                                     labels,
                                     out_index=index,
                                     device=self.data_device,
                                     escape=type(self.cache.g))
        return sequence

    def config_test_data(self, index):
        labels = self.graph.label[index]
        sequence = FullBatchSequence([self.cache.feat, self.cache.g],
                                     labels,
                                     out_index=index,
                                     device=self.device,
                                     escape=type(self.cache.g))
        return sequence

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.01)
        weight_decay = self.cfg.get('weight_decay', 5e-6)
        return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

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

        binary_reg = self.cfg.get("binary_reg", 0.)

        for epoch, batch in enumerate(dataloader):
            self.callbacks.on_train_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)

            if not isinstance(x, tuple):
                x = x,
            out, z = model(*x)
            if out_index is not None:
                out = out[out_index]
            loss = loss_fn(out, y) + torch.norm(z * (torch.ones_like(z) - z), p=1) * binary_reg
            loss.backward()
            for metric in self.metrics:
                metric.update_state(y.cpu(), out.detach().cpu())
            self.callbacks.on_train_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))
