import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import Sequential
from graphgallery.nn.metrics.pytorch import Accuracy


class Mlp(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features, bias=bias)
        self.fc2 = nn.Linear(out_features, out_features, bias=bias)
        self.act_fn = torch.nn.functional.gelu
        self.reset_parameters()

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(out_features, eps=1e-6)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x @ x.T
    mask = torch.eye(x_dis.shape[0], device=x.device)
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum**(-1))
    x_dis = (1 - mask) * x_dis
    return x_dis

def Ncontrast(x_dis, adj_label, tau=1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp( tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))).mean()
    return loss

class GMLP(TorchKeras):
    def __init__(self,
                 in_features,
                 out_features,
                 *,
                 tau=2,
                 alpha=10.0,
                 hids=[256],
                 acts=None,
                 dropout=0.6,
                 weight_decay=5e-3,
                 lr=0.001,
                 bias=True):

        super().__init__()
        mlp = []
        for hid in hids:
            mlp.append(Mlp(in_features, hid, dropout, bias=bias))
            in_features = hid
        self.mlp = Sequential(*mlp)
        self.classifier = nn.Linear(in_features, out_features, bias=bias)
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(self.parameters(),
                                          weight_decay=weight_decay,
                                          lr=lr),
                     metrics=[Accuracy()])
        self.tau = tau
        self.alpha = alpha

    def forward(self, x):
        x = self.mlp(x)

        feature_cls = x
        Z = x

        if self.training:
            x_dis = get_feature_dis(Z)

        out = self.classifier(feature_cls)
#         out = F.log_softmax(out, dim=1)

        if self.training:
            return out, x_dis
        else:
            return out
        
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
        x = [_x.to(device) if hasattr(_x, 'to') else _x for _x in x]
        y = y.to(device)
        
        out, x_dis = self(x[0])
        if out_weight is not None:
            out = out[out_weight]
            
        loss = loss_fn(out, y) + Ncontrast(x_dis, x[1], tau=self.tau) * self.alpha
        loss.backward()
        optimizer.step()
        for metric in metrics:
            metric.update_state(y.cpu(), out.detach().cpu())

        results = [loss.cpu().detach()] + [metric.result() for metric in metrics]
        return dict(zip(self.metrics_names, results))        
