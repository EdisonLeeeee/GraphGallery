import torch
from graphgallery.nn.models.torch_keras import TorchKeras, to_device


class AutoEncoder(TorchKeras):

    def encode(self, x, adj):
        return self.encoder(x, adj)

    def decode(self, z, edge_index=None):
        return self.decoder(z, edge_index=edge_index)

    def train_step_on_batch(self,
                            x,
                            y,
                            out_index=None,
                            device="cpu"):
        self.train()
        optimizer = self.optimizer
        loss_fn = self.loss
        metrics = self.metrics
        optimizer.zero_grad()
        x, y = to_device(x, y, device=device)
        out = self(*x)
        if out_index is not None:
            out = out[out_index[0], out_index[1]]
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
            
        # HERE we dont not update metrics in training to save time
#         for metric in metrics:
#             metric.update_state(y.cpu(), out.detach().cpu())

        results = [loss.cpu().detach()] + [metric.result() for metric in metrics]
        return dict(zip(self.metrics_names, results))

    @torch.no_grad()
    def test_step_on_batch(self,
                           x,
                           y,
                           out_index=None,
                           device="cpu"):
        self.eval()
        loss_fn = self.loss
        metrics = self.metrics
        x, y = to_device(x, y, device=device)
        z = self.encode(*x)
        out = self.decode(z, out_index)
        loss = loss_fn(out, y)
        for metric in metrics:
            metric.update_state(y.cpu(), out.detach().cpu())

        results = [loss.cpu().detach()] + [metric.result() for metric in metrics]
        return dict(zip(self.metrics_names, results))

    @torch.no_grad()
    def predict_step_on_batch(self, x, out_index=None, device="cpu"):
        self.eval()
        x = to_device(x, device=device)
        z = self.encode(*x)
        out = self.decode(z, out_index)
        return out.cpu().detach()
