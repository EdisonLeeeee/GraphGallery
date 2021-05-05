import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.metrics.pytorch import Accuracy
from graphgallery.nn.layers.pytorch import activations, SpectralEigenConv


class SAT(TorchKeras):
    def __init__(self,
                 in_features,
                 out_features,
                 K=10,
                 alpha=0.1,
                 eps_U=0.3,
                 eps_V=1.2,
                 lamb_U=0.8,
                 lamb_V=0.8,
                 hids=[],
                 acts=[],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=False):

        super().__init__()

        lin = []
        lin.append(nn.Dropout(dropout))

        for hid, act in zip(hids, acts):
            lin.append(nn.Linear(in_features,
                                 hid,
                                 bias=bias))
            lin.append(activations.get(act))
            lin.append(nn.Dropout(dropout))
            in_features = hid
        lin = nn.Sequential(*lin)
        conv = SpectralEigenConv(in_features, out_features, bias=bias, K=K, alpha=alpha)

        self.lin = lin
        self.conv = conv
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(self.parameters(),
                                          weight_decay=weight_decay, lr=lr),
                     metrics=[Accuracy()])
        self.eps_U = eps_U
        self.eps_V = eps_V
        self.lamb_U = lamb_U
        self.lamb_V = lamb_V

    def forward(self, x, U, V=None):
        """
        x: node attribute matrix
        if `V=None`:
            U: (N, N) adjacency matrix
        else:
            U: (N, k) eigenvector matrix
            V: (k,) eigenvalue
        """
        x = self.lin(x)
        x = self.conv(x, U, V)
        return x

#     def train_step_on_batch(self,
#                             x,
#                             y,
#                             out_weight=None,
#                             device="cpu"):
#         self.train()
#         optimizer = self.optimizer
#         loss_fn = self.loss
#         metrics = self.metrics
#         optimizer.zero_grad()

#         if not isinstance(x, (list, tuple)):
#             x = [x]
#         x = [_x.to(device) if hasattr(x, 'to') else _x for _x in x]
#         y = y.to(device)

#         ########################
#         x, U, V = x
#         U.requires_grad = True
#         V.requires_grad = True
#         out = self(x, U, V)
#         if out_weight is not None:
#             out = out[out_weight]
#         loss = loss_fn(out, y)

#         U_grad, V_grad = torch.autograd.grad(loss, [U, V], retain_graph=True)
#         U.requires_grad = False
#         V.requires_grad = False

#         U_grad = self.eps_U * U_grad / torch.norm(U_grad)
#         V_grad = self.eps_V * V_grad / torch.norm(V_grad)

#         out_U = self(x, U + U_grad, V)
#         out_V = self(x, U, V + V_grad)

#         if out_weight is not None:
#             out_U = out_U[out_weight]
#             out_V = out_V[out_weight]

#         loss += self.lamb_U * loss_fn(out_U, y) + self.lamb_V * loss_fn(out_V, y)
#         ########################

#         loss.backward()
#         optimizer.step()
#         for metric in metrics:
#             metric.update_state(y.cpu(), out.detach().cpu())

#         results = [loss.cpu().detach()] + [metric.result() for metric in metrics]
#         return dict(zip(self.metrics_names, results))
