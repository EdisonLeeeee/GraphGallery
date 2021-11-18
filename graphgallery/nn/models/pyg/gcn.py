import torch.nn as nn
import torch.nn.functional as F
from graphgallery.nn.layers.pytorch import Sequential, activations

from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj


class GCN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 bias=True):

        super().__init__()

        conv = []
        conv.append(nn.Dropout(dropout))
        for hid, act in zip(hids, acts):
            conv.append(GCNConv(in_features,
                                hid,
                                cached=True,
                                bias=bias,
                                normalize=False))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid
        conv.append(GCNConv(in_features,
                            out_features,
                            cached=True,
                            bias=bias,
                            normalize=False))
        conv = Sequential(*conv)

        self.conv = conv
        self.reg_paras = conv[1].parameters()
        self.non_reg_paras = conv[2:].parameters()

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight)


class DropEdge(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 p=0.3,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=True):

        super().__init__()

        conv = []
        conv.append(nn.Dropout(dropout))
        for hid, act in zip(hids, acts):
            conv.append(GCNConv(in_features,
                                hid,
                                cached=True,
                                bias=bias,
                                normalize=False))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid
        conv.append(GCNConv(in_features,
                            out_features,
                            cached=True,
                            bias=bias,
                            normalize=False))
        conv = Sequential(*conv)

        self.p = p
        self.conv = conv

    def forward(self, x, edge_index, edge_weight=None):
        if self.training and self.p:
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=self.p)
        return self.conv(x, edge_index, edge_weight)


class RDrop(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 p=0.3,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 kl=0.005,
                 lr=0.01,
                 bias=True):

        super().__init__()

        conv = []
        conv.append(nn.Dropout(dropout))
        for hid, act in zip(hids, acts):
            conv.append(GCNConv(in_features,
                                hid,
                                cached=True,
                                bias=bias,
                                normalize=False))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid
        conv.append(GCNConv(in_features,
                            out_features,
                            cached=True,
                            bias=bias,
                            normalize=False))
        conv = Sequential(*conv)

        self.p = p
        self.kl = kl
        self.conv = conv
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam([dict(params=conv[1].parameters(),
                                                weight_decay=weight_decay),
                                           dict(params=conv[2:].parameters(),
                                                weight_decay=0.)], lr=lr),
                     metrics=[Accuracy()])

    def forward(self, x, edge_index, edge_weight=None):
        if self.training and self.p:
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=self.p)
        return self.conv(x, edge_index, edge_weight)

    def forward_step(self, x, out_index=None):
        z = self(*x)
        z2 = self(*x)

        # index select or mask outputs
        pred = self.index_select(z, out_index=out_index)
        pred2 = self.index_select(z2, out_index=out_index)

        return dict(z=z, pred=pred, pred2=pred2)

    def compute_loss(self, output_dict, y):
        if self.training:
            loss1 = self.loss(output_dict['pred'], y)
            loss2 = self.loss(output_dict['pred2'], y)
            loss_kl = self.compute_kl_loss(output_dict['pred'], output_dict['pred2'])
            loss = 0.5 * (loss1 + loss2) + self.kl * loss_kl
            return loss
        else:
            return self.loss(output_dict['pred'], y)

    @staticmethod
    def compute_kl_loss(p, q):

        q = F.softmax(q, dim=-1)
        p = F.softmax(p, dim=-1)

        p_loss = F.kl_div(p.log(), q, reduction='none')
        q_loss = F.kl_div(q.log(), p, reduction='none')

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss
