import torch
import torch.nn as nn

from graphgallery.nn.layers.pytorch import GCNConv, Sequential, activations


class LATGCN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 num_nodes,
                 *,
                 eta=0.1,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.2,
                 bias=False):
        super().__init__()
        assert hids, "LATGCN requires hidden layers"
        conv = []
        conv.append(nn.Dropout(dropout))
        for hid, act in zip(hids, acts):
            conv.append(GCNConv(in_features,
                                hid,
                                bias=bias))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid
        conv.append(GCNConv(in_features, out_features, bias=bias))
        conv = Sequential(*conv)

        self.zeta = nn.Parameter(torch.randn(num_nodes, hids[0]))
        self.conv1 = conv[:3]  # includes dropout, ReLU and the first GCN layer
        self.conv2 = conv[3:]  # the remaining

        self.eta = eta

        self.reg_paras = self.conv1.parameters()
        self.non_reg_paras = self.conv2.parameters()

    def forward(self, x, adj):
        h = self.conv1(x, adj)
        logit = self.conv2(h, adj)

        if self.training:
            zeta = clip_by_norm(self.zeta, self.eta)
            hp = h + zeta
            logitp = self.conv2(hp, adj)
            reg_loss = torch.norm(logitp - logit)
            return logit, reg_loss
        else:
            return logit


# @torch.no_grad()
def clip_by_norm(tensor, clip_norm):
    l2_norm = torch.norm(tensor, p=2, dim=1).view(-1, 1)
    tensor = tensor * clip_norm / l2_norm
    return tensor
