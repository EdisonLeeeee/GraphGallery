import torch
from graphgallery.nn.models import TorchKeras


class BaseSAT(TorchKeras):
    def train_step_on_batch(self,
                            x,
                            y,
                            out_weight=None,
                            device="cpu"):
        self.train()
        optimizer = self.optimizer
        loss_fn = self.loss
        metrics = self.metrics
        optimizer.zero_grad()

        if not isinstance(x, (list, tuple)):
            x = [x]
        x = [_x.to(device) if hasattr(x, 'to') else _x for _x in x]
        y = y.to(device)

        ########################
        x, U, V = x
        U.requires_grad = True
        V.requires_grad = True
        out = self(x, U, V)
        if out_weight is not None:
            out = out[out_weight]
        loss = loss_fn(out, y)

        U_grad, V_grad = torch.autograd.grad(loss, [U, V], retain_graph=True)
        U.requires_grad = False
        V.requires_grad = False

        U_grad = self.eps_U * U_grad / torch.norm(U_grad, 2)
        V_grad = self.eps_V * V_grad / torch.norm(V_grad, 2)

        out_U = self(x, U + U_grad, V)
        out_V = self(x, U, V + V_grad)

        if out_weight is not None:
            out_U = out_U[out_weight]
            out_V = out_V[out_weight]

        loss += self.lamb_U * loss_fn(out_U, y) + self.lamb_V * loss_fn(out_V, y)
        ########################

        loss.backward()
        optimizer.step()
        for metric in metrics:
            metric.update_state(y.cpu(), out.detach().cpu())

        results = [loss.cpu().detach()] + [metric.result() for metric in metrics]
        return dict(zip(self.metrics_names, results))
