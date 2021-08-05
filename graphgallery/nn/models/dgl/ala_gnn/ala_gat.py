import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.models.torch_keras import to_device
from graphgallery.nn.metrics.pytorch import Accuracy

from graphgallery.nn.layers.dgl import GatedAttnLayer
from dgl.nn.pytorch import GATConv


class ALaGAT(TorchKeras):
    def __init__(
        self,
        in_features,
        out_features,
        num_nodes,
        num_heads,
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

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hids[0])))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hids[0])))

        conv = []
        for ix, (hid, act) in enumerate(zip(hids, acts)):
            if ix == 0:
                layer = GATConv(
                    in_features,
                    hid,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                )
            else:
                layer = GatedAttnLayer(
                    in_features,
                    hid,
                    num_nodes,
                    num_heads,
                    bias=bias,
                    activation=act,
                    share_tau=share_tau,
                    lidx=ix,
                )

            conv.append(layer)
            in_features = hid

        self.conv = nn.ModuleList(conv)
        self.lin = nn.Linear(in_features * num_heads, out_features, bias=bias)

        self.compile(
            loss=nn.CrossEntropyLoss(),
            optimizer=optim.AdamW(self.parameters(), weight_decay=weight_decay, lr=lr),
            metrics=[Accuracy()],
        )
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.conv:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        self.lin.reset_parameters()
        nn.init.constant_(self.global_tau1, 0.5)
        nn.init.constant_(self.global_tau2, 0.5)

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

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
        return

    def forward(self, x, g):

        z = torch.FloatTensor(
            [
                1.0,
            ]
        ).to(x.device)
        h = x
        list_z = []
        for lidx, layer in enumerate(self.conv):
            if lidx == 0:
                # first layer use initial weight_y
                logits = F.softmax(torch.mm(h, self.init_weight_y), dim=1)
                h = layer(g, h)
            else:
                logits = F.softmax(self.lin(h.reshape(h.size(0), -1)), dim=1)
                h, z = layer(
                    g,
                    h,
                    logits,
                    old_z=z,
                    attn_l=self.attn_l,
                    attn_r=self.attn_r,
                    tau1=self.global_tau1,
                    tau2=self.global_tau2,
                )
                h = self.dropout(h)
                list_z.append(z.flatten())

        out = self.lin(h.reshape(h.size(0), -1))
        if self.training:
            all_z = torch.stack(list_z, dim=1)  # (n_nodes, n_layers)
            return out, all_z
        else:
            return out

    def train_step_on_batch(self, x, y, out_index=None, device="cpu"):
        self.train()
        optimizer = self.optimizer
        loss_fn = self.loss
        metrics = self.metrics
        optimizer.zero_grad()
        x, y = to_device(x, y, device=device)
        out, z = self(*x)
        if out_index is not None:
            out = out[out_index]
        loss = (
            loss_fn(out, y)
            + torch.norm(z * (torch.ones_like(z) - z), p=1) * self.binary_reg
        )
        loss.backward()
        optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        for metric in metrics:
            metric.update_state(y.cpu(), out.detach().cpu())

        results = [loss.cpu().detach()] + [metric.result() for metric in metrics]
        return dict(zip(self.metrics_names, results))
