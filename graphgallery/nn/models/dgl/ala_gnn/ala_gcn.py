import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from graphgallery.nn.models import TorchEngine
from graphgallery.nn.metrics.pytorch import Accuracy
from graphgallery.nn.layers.dgl import GatedLayer

from dgl.nn.pytorch import GraphConv


class ALaGCN(TorchEngine):
    def __init__(
        self,
        in_features,
        out_features,
        num_nodes,
        *,
        binary_reg=0.0,
        hids=[16] * 2,
        acts=[None] * 2,
        dropout=0.5,
        weight_decay=5e-6,
        lr=0.01,
        bias=False,
        share_tau=True,
    ):

        super().__init__()
        self.out_features = out_features
        self.binary_reg = binary_reg

        self.dropout = nn.Dropout(dropout)
        self.global_tau1 = nn.Parameter(torch.tensor(0.5))
        self.global_tau2 = nn.Parameter(torch.tensor(0.5))

        conv = []
        for ix, (hid, act) in enumerate(zip(hids, acts)):
            if ix == 0:
                layer = GraphConv(in_features, hid, bias=bias, activation=act)
            else:
                layer = GatedLayer(in_features, hid,
                                   num_nodes,
                                   bias=bias,
                                   activation=act,
                                   share_tau=share_tau,
                                   lidx=ix,
                                   )

            conv.append(layer)
            in_features = hid

        self.conv = nn.ModuleList(conv)
        self.lin = nn.Linear(in_features, out_features, bias=bias)

        self.compile(
            loss=nn.CrossEntropyLoss(),
            optimizer=optim.AdamW(self.parameters(), weight_decay=weight_decay, lr=lr),
            metrics=[Accuracy()],
        )

    def reset_parameters(self):
        for conv in self.conv:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        self.lin.reset_parameters()
        nn.init.constant_(self.global_tau1, 1 / 2)
        nn.init.constant_(self.global_tau2, 1 / 2)

    def on_train_begin(self):
        feats = self.cache["x"]
        label = self.cache["y"]
        # initial weight_y is obtained by linear regression
        A = torch.mm(feats.t(), feats) + 1e-05 * torch.eye(
            feats.size(1), device=feats.device
        )
        # (feats, feats)
        labels_one_hot = torch.zeros(
            (feats.size(0), self.out_features), device=feats.device
        )
        labels_one_hot[torch.arange(label.size(0)), label] = 1
        self.init_weight_y = torch.mm(
            torch.mm(torch.cholesky_inverse(A), feats.t()), labels_one_hot
        )

    def forward(self, x, g):

        z = torch.FloatTensor([1.0, ]).to(x.device)
        h = x
        list_z = []
        for lidx, layer in enumerate(self.conv):
            if lidx == 0:
                # first layer use initial weight_y
                logits = F.softmax(torch.mm(h, self.init_weight_y), dim=1)
                h = layer(g, h)
            else:
                logits = F.softmax(self.lin(h), dim=1)
                h, z = layer(
                    g, h, logits, old_z=z, tau1=self.global_tau1, tau2=self.global_tau2
                )
                h = self.dropout(h)
                list_z.append(z)

        z = self.lin(h)
        z_stack = torch.stack(list_z, dim=1)  # (n_nodes, n_layers)
        return dict(z=z, z_stack=z_stack)

    def compute_loss(self, output_dict, y):
        pred = output_dict['pred']
        loss = self.loss(pred, y)

        if self.training:
            z = output_dict['z_stack']
            loss += torch.norm(z * (torch.ones_like(z) - z), p=1) * self.binary_reg

        return loss
