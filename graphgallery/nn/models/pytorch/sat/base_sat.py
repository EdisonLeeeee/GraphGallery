import torch
from graphgallery.nn.models.torch_keras import TorchKeras, to_device


class BaseSAT(TorchKeras):
    def train_step_on_batch(self,
                            x,
                            y,
                            out_index=None,
                            device="cpu"):
        self.train()
        optimizer = self.optimizer
        loss_fn = self.loss
        optimizer.zero_grad()
        x, y = to_device(x, y, device=device)

        ########################
        x, U, V = x
        U.requires_grad = True
        V.requires_grad = True
        out = self(x, U, V)
        if out_index is not None:
            out = out[out_index]
        loss = loss_fn(out, y)

        U_grad, V_grad = torch.autograd.grad(loss, [U, V], retain_graph=True)
        U.requires_grad = False
        V.requires_grad = False

        U_grad = self.eps_U * U_grad / torch.norm(U_grad, 2)
        V_grad = self.eps_V * V_grad / torch.norm(V_grad, 2)

        out_U = self(x, U + U_grad, V)
        out_V = self(x, U, V + V_grad)

        if out_index is not None:
            out_U = out_U[out_index]
            out_V = out_V[out_index]

        loss += self.lamb_U * loss_fn(out_U, y) + self.lamb_V * loss_fn(out_V, y)
        ########################

        loss.backward()
        optimizer.step()
        self.update_metrics(out, y)

        results = [loss.cpu().detach()] + [metric.result() for metric in self.metrics]
        return dict(zip(self.metrics_names, results))
