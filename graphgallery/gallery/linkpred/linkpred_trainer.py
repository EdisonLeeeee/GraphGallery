import torch
from torch import Tensor
from typing import Optional, Union, Any, Callable, List

from torch.utils.data import DataLoader, Dataset

from graphgallery.gallery import Trainer
from graphgallery.nn.metrics import AveragePrecision, AUC
from graphgallery.nn.losses import BCELoss
from graphgallery.functional import negative_sampling


class LinkPredTrainer(Trainer):
    def train_step(self, dataloader: DataLoader) -> dict:
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

        for epoch, batch in enumerate(dataloader):
            self.callbacks.on_train_batch_begin(epoch)
            x, _, out_index = self.unravel_batch(batch)
            x = self.to_device(x)

            if not isinstance(x, tuple):
                x = x,

            z = model(*x)

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

            loss = loss_fn(out, y)
            loss.backward()
            for metric in self.metrics:
                metric.update_state(y.cpu(), out.detach().cpu())
            self.callbacks.on_train_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))

    @torch.no_grad()
    def test_step(self, dataloader: DataLoader) -> dict:
        loss_fn = self.loss
        model = self.model
        model.eval()
        callbacks = self.callbacks
        self.reset_metrics()

        for epoch, batch in enumerate(dataloader):
            callbacks.on_test_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)
            if not isinstance(x, tuple):
                x = x,
            z = model(*x)
            out = model.decoder(z, out_index)
            loss = loss_fn(out, y)

            for metric in self.metrics:
                metric.update_state(y.cpu(), out.detach().cpu())
            callbacks.on_test_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))

    @torch.no_grad()
    def predict_step(self, dataloader: DataLoader) -> Tensor:
        model = self.model
        model.eval()
        outs = []
        callbacks = self.callbacks
        for epoch, batch in enumerate(dataloader):
            callbacks.on_predict_batch_begin(epoch)
            x, _, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            if not isinstance(x, tuple):
                x = x,
            out = model(*x)
            z = model(*x)
            out = model.decoder(z, out_index)

            outs.append(out)
            callbacks.on_predict_batch_end(epoch)

        return torch.cat(outs, dim=0)

    def predict(self, predict_data=None,
                transform: Callable = torch.nn.Sigmoid()) -> Tensor:

        if not self.model:
            raise RuntimeError(
                'You must compile your model before training/testing/predicting. Use `trainer.build()`.'
            )

        if not isinstance(predict_data, (DataLoader, Dataset)):
            predict_data = self.config_predict_data(predict_data)

        out = self.predict_step(predict_data).squeeze()
        if transform is not None:
            out = transform(out)
        return out

    def config_metrics(self) -> List:
        return [AUC(), AveragePrecision()]

    def config_loss(self) -> Callable:
        return BCELoss()

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.01)
        weight_decay = self.cfg.get('weight_decay', 0.)
        return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
