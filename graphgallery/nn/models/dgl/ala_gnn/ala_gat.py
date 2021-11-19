import torch
import torch.nn as nn
import torch.nn.functional as F
from graphgallery.nn.layers.dgl import GatedAttnLayer
from dgl.nn.pytorch import GATConv


class ALaGAT(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_nodes,
        num_heads,
        *,
        hids=[8] * 2,
        acts=[None] * 2,
        dropout=0.5,
        bias=False,
        share_tau=True,
    ):

        super().__init__()
        assert len(hids) > 1

        conv = []
        for ix, (hid, act) in enumerate(zip(hids, acts)):
            if ix == 0:
                layer = GATConv(in_features, hid,
                                num_heads=num_heads,
                                feat_drop=dropout,
                                attn_drop=dropout)
            else:
                layer = GatedAttnLayer(in_features, hid,
                                       num_nodes,
                                       num_heads,
                                       bias=bias,
                                       activation=act,
                                       share_tau=share_tau,
                                       lidx=ix)

            conv.append(layer)
            in_features = hid

        self.conv = nn.ModuleList(conv)
        self.lin = nn.Linear(in_features * num_heads, out_features, bias=bias)
        self.init_weight_y = None
        self.dropout = nn.Dropout(dropout)
        self.global_tau1 = nn.Parameter(torch.tensor(0.5))
        self.global_tau2 = nn.Parameter(torch.tensor(0.5))

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hids[0])))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hids[0])))

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
                logits = F.softmax(self.lin(h.reshape(h.size(0), -1)), dim=1)
                h, z = layer(g, h,
                             logits,
                             old_z=z,
                             attn_l=self.attn_l,
                             attn_r=self.attn_r,
                             tau1=self.global_tau1,
                             tau2=self.global_tau2,
                             )
                h = self.dropout(h)
                list_z.append(z.flatten())

        z = self.lin(h.view(h.size(0), -1))

        if self.training:
            z_stack = torch.stack(list_z, dim=1)  # (n_nodes, n_layers)
            return z, z_stack
        else:
            return z
