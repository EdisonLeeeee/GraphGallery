import torch
import graphgallery.nn.models.pytorch as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery.nodeclas import NodeClasTrainer


@PyTorch.register()
class LATGCN(NodeClasTrainer):
    """
        Implementation of Latent Adversarial Training of Graph Convolutional Networks (LATGCN).
        `Latent Adversarial Training of Graph Convolutional Networks
        <https://graphreason.github.io/papers/35.pdf>`
        Tensorflow 1.x implementation: <https://github.com/cshjin/LATGCN/>

    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  feat_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)

        feat, adj = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``adj`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, adj=adj)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.2,
                   bias=False,
                   eta=0.1):

        assert hids, "LATGCN requires hidden layers"

        model = models.LATGCN(self.graph.num_feats,
                              self.graph.num_classes,
                              self.graph.num_nodes,
                              eta=eta,
                              hids=hids,
                              acts=acts,
                              dropout=dropout,
                              bias=bias)

        return model

    def config_train_data(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.feat, self.cache.adj],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.01)
        weight_decay = self.cfg.get('weight_decay', 5e-4)
        model = self.model
        self.zeta_opt = torch.optim.Adam([model.zeta], lr=lr)
        return torch.optim.Adam([dict(params=model.reg_paras,
                                      weight_decay=weight_decay),
                                 dict(params=model.non_reg_paras,
                                      weight_decay=0.)], lr=lr)

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

        zeta_opt = self.zeta_opt
        zeta = model.zeta
        gamma = self.cfg.get("gamma", 0.01)
        inner_epochs = self.cfg.get("inner_epochs", 20)

        for epoch, batch in enumerate(dataloader):
            self.callbacks.on_train_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)
            if not isinstance(x, tuple):
                x = x,

            for _ in range(inner_epochs):
                zeta_opt.zero_grad()
                _, reg_loss = model(*x)
                reg_loss = -reg_loss
                zeta.grad = torch.autograd.grad(reg_loss, zeta)[0]
                zeta_opt.step()

            out, reg_loss = model(*x)
            if out_index is not None:
                out = out[out_index]
            loss = loss_fn(out, y) + gamma * reg_loss
            loss.backward()
            for metric in self.metrics:
                metric.update_state(y.cpu(), out.detach().cpu())
            self.callbacks.on_train_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))
