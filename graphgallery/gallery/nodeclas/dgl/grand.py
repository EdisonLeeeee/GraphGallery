import torch
import graphgallery.nn.models.dgl as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import NodeClasTrainer

from graphgallery.gallery.nodeclas import DGL


@DGL.register()
class GRAND(NodeClasTrainer):
    """
        Implementation of GRAND. 
        `Graph Random Neural Network for Semi-Supervised Learning on Graphs  
        <https://arxiv.org/pdf/2005.11079.pdf>`
        Pytorch implementation: <https://github.com/THUDM/GRAND>
    """

    def data_step(self,
                  adj_transform="add_self_loop",
                  feat_transform=None):
        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)
        feat, g = gf.astensors(attr_matrix, adj_matrix,
                               device=self.data_device)

        # ``g`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, g=g)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   S=1,
                   K=4,
                   dropout=0.5,
                   bias=False,
                   bn=False):

        model = models.GRAND(self.graph.num_feats,
                             self.graph.num_classes,
                             hids=hids,
                             acts=acts,
                             S=S,
                             K=K,
                             dropout=dropout,
                             bias=bias,
                             bn=bn)

        return model

    def config_train_data(self, index):
        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.feat, self.cache.g],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device,
                                     escape=type(self.cache.g))
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
        temp = self.cfg.get("temp", 0.5)
        lam = self.cfg.get("lam", 1.0)

        for epoch, batch in enumerate(dataloader):
            self.callbacks.on_train_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)

            if not isinstance(x, tuple):
                x = x,
            outs = zs = model(*x)
            if out_index is not None:
                outs = tuple(out[out_index] for out in outs)
            loss = 0.
            for out in outs:
                loss += loss_fn(out, y)
            loss /= len(outs)
            loss += consis_loss(zs, temp=temp, lam=lam)
            loss.backward()
            for metric in self.metrics:
                metric.update_state(y.cpu(), outs[0].detach().cpu())
            self.callbacks.on_train_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))


def consis_loss(logps, temp=0.5, lam=1.):
    ps = [torch.exp(p) for p in logps]
    ps = torch.stack(ps, dim=2)

    avg_p = torch.mean(ps, dim=2)
    sharp_p = (torch.pow(avg_p, 1. / temp) /
               torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()

    sharp_p = sharp_p.unsqueeze(2)
    loss = torch.mean(
        torch.sum(torch.pow(ps - sharp_p, 2), dim=1, keepdim=True))

    loss = lam * loss
    return loss
