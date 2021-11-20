import torch
import scipy.sparse as sp
import graphgallery.nn.models.pytorch as models

from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery.nodeclas import NodeClasTrainer


@PyTorch.register()
class SATGCN(NodeClasTrainer):
    def data_step(self,
                  adj_transform="normalize_adj",
                  feat_transform=None,
                  k=35):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)

        V, U = sp.linalg.eigsh(adj_matrix, k=k)

        adj_matrix = (U * V) @ U.T
        adj_matrix[adj_matrix < 0] = 0.
        adj_matrix = gf.get(adj_transform)(adj_matrix)

        feat, adj, U, V = gf.astensors(attr_matrix,
                                       adj_matrix,
                                       U,
                                       V,
                                       device=self.data_device)

        # ``adj`` , ``feat`` , U`` and ``V`` are cached for later use
        self.register_cache(feat=feat, adj=adj, U=U, V=V)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.5,
                   bias=True):

        model = models.sat.GCN(self.graph.num_feats,
                               self.graph.num_classes,
                               hids=hids,
                               acts=acts,
                               dropout=dropout,
                               bias=bias)

        return model

    def config_train_data(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.feat, self.cache.U, self.cache.V],
                                     y=labels,
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

    def train_step(self, dataloader) -> dict:

        loss_fn = self.loss
        model = self.model

        self.reset_metrics()
        model.train()

        eps_U = self.cfg.get("eps_U", 0.1)
        eps_V = self.cfg.get("eps_V", 0.1)
        lamb_U = self.cfg.get("lamb_U", 0.5)
        lamb_V = self.cfg.get("lamb_V", 0.5)

        for epoch, batch in enumerate(dataloader):
            self.callbacks.on_train_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)

            # =================== Spectral Adversarial Training here===========
            feat, U, V = x
            U.requires_grad_()
            V.requires_grad_()

            out = model(feat, U, V)

            if out_index is not None:
                out = out[out_index]

            loss = loss_fn(out, y)
            U_grad, V_grad = torch.autograd.grad(loss, [U, V], retain_graph=True)

            U.requires_grad_(False)
            V.requires_grad_(False)

            U_grad = eps_U * U_grad / torch.norm(U_grad, 2)
            V_grad = eps_V * V_grad / torch.norm(V_grad, 2)

            out_U = model(feat, U + U_grad, V)
            out_V = model(feat, U, V + V_grad)

            if out_index is not None:
                out_U = out_U[out_index]
                out_V = out_V[out_index]

            loss += lamb_U * loss_fn(out_U, y) + lamb_V * loss_fn(out_V, y)
            # ===============================================================

            loss.backward()
            for metric in self.metrics:
                metric.update_state(y.cpu(), out.detach().cpu())
            self.callbacks.on_train_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))


@PyTorch.register()
class SATSGC(SATGCN):
    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   K=2,
                   dropout=0.5,
                   bias=True):

        model = models.sat.SGC(self.graph.num_feats,
                               self.graph.num_classes,
                               K=K,
                               hids=hids,
                               acts=acts,
                               dropout=dropout,
                               bias=bias)

        return model


@PyTorch.register()
class SATSSGC(SATGCN):
    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   K=5,
                   alpha=0.1,
                   dropout=0.5,
                   bias=True):

        model = models.sat.SSGC(self.graph.num_feats,
                                self.graph.num_classes,
                                K=K,
                                alpha=alpha,
                                hids=hids,
                                acts=acts,
                                dropout=dropout,
                                bias=bias)

        return model
