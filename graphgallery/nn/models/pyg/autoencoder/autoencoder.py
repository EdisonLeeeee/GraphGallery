import torch
from graphgallery.nn.models.torch_keras import TorchKeras, to_device
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)


class AutoEncoder(TorchKeras):

    def encode(self, x, edge_index, edge_weight=None):
        return self.encoder(x, edge_index, edge_weight)

    def decode(self, z, edge_index=None):
        return self.decoder(z, edge_index=edge_index)

    def train_step_on_batch(self,
                            x,
                            y=None,
                            out_index=None,
                            device="cpu"):
        self.train()
        optimizer = self.optimizer
        loss_fn = self.loss
        metrics = self.metrics
        optimizer.zero_grad()
        x = to_device(x, device=device)
        z = self.encode(*x)
        # here `out_index` maybe pos_edge_index
        # or (pos_edge_index, neg_edge_index)
        if isinstance(out_index, (list, tuple)):
            assert len(out_index) == 2, '`out_index` should be (pos_edge_index, neg_edge_index) or pos_edge_index'
            pos_edge_index, neg_edge_index = out_index
        else:
            pos_edge_index, neg_edge_index = out_index, None

        pos_pred = self.decode(z, pos_edge_index)
        pos_y = z.new_ones(pos_edge_index.size(1))

        # Do not include self-loops in negative samples
        # TODO: is it necessary?
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_pred = self.decode(z, neg_edge_index)

        loss = loss_fn(pos_pred, neg_pred)
        loss.backward()
        optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        for metric in metrics:
            metric.update_state(y.cpu(), pred.detach().cpu())

        results = [loss.cpu().detach()] + [metric.result() for metric in metrics]
        return dict(zip(self.metrics_names, results))

    @torch.no_grad()
    def test_step_on_batch(self,
                           x,
                           y,
                           out_index=None,
                           device="cpu"):
        self.eval()
        metrics = self.metrics
        x = to_device(x, device=device)
        z = self.encode(*x)
        pred = self.decode(z, out_index)

        for metric in metrics:
            metric.update_state(y.cpu(), pred.detach().cpu())

        results = [None] + [metric.result() for metric in metrics]
        return dict(zip(self.metrics_names, results))

    @torch.no_grad()
    def predict_step_on_batch(self, x, out_index=None, device="cpu"):
        self.eval()
        x = to_device(x, device=device)
        z = self.encode(*x)
        pred = self.decode(z, out_index)
        return pred.cpu().detach()
