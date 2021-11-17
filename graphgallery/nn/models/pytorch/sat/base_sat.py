import torch


class BaseSAT(torch.nn.Module):
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
        z = self(x, U, V)
        pred = self.index_select(z, out_index=out_index)
        loss = loss_fn(pred, y)

        U_grad, V_grad = torch.autograd.grad(loss, [U, V], retain_graph=True)
        U.requires_grad = False
        V.requires_grad = False

        U_grad = self.eps_U * U_grad / torch.norm(U_grad, 2)
        V_grad = self.eps_V * V_grad / torch.norm(V_grad, 2)

        out_U = self(x, U + U_grad, V)
        out_V = self(x, U, V + V_grad)

        out_U = self.index_select(out_U, out_index=out_index)
        out_V = self.index_select(out_V, out_index=out_index)

        loss += self.lamb_U * loss_fn(out_U, y) + self.lamb_V * loss_fn(out_V, y)
        ########################

        loss.backward()
        optimizer.step()
        metrics = self.compute_metrics(dict(pred=pred), y)

        results = [loss.cpu().detach()] + metrics
        return dict(zip(self.metrics_names, results))
