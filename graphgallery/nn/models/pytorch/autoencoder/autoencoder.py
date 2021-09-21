import torch
from graphgallery.nn.models.torch_keras import TorchKeras, to_device


class AutoEncoder(TorchKeras):

    def encode(self, x, adj):
        return self.encoder(x, adj)

    def decode(self, z, edge_index=None):
        return self.decoder(z, edge_index=edge_index)

    def update_metrics(self, out, y):
        # HERE we dont not update metrics in training to save time
        if self.training:
            return
        for metric in self.metrics:
            metric.update_state(y.cpu(), out.detach().cpu())

    @torch.no_grad()
    def test_step_on_batch(self,
                           x,
                           y,
                           out_index=None,
                           device="cpu"):
        self.eval()
        x, y = to_device(x, y, device=device)
        z = self.encode(*x)
        out = self.decode(z, out_index)
        loss = self.compute_loss(out, y)
        self.update_metrics(out, y)

        if loss is not None:
            loss = loss.cpu().item()

        results = [loss] + [metric.result() for metric in self.metrics]
        return dict(zip(self.metrics_names, results))

    @torch.no_grad()
    def predict_step_on_batch(self, x, out_index=None, device="cpu"):
        self.eval()
        x = to_device(x, device=device)
        z = self.encode(*x)
        out = self.decode(z, out_index)
        return out.cpu().detach()
