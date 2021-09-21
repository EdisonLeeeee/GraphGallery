import torch
import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.metrics.pytorch import Accuracy
from graphgallery.nn.layers.pytorch import activations

import dgl.function as fn
from dgl.nn.pytorch.conv import GraphConv


class JKNet(TorchKeras):
    def __init__(self,
                 in_features, out_features, *,
                 hids=[16] * 5, acts=['relu'] * 5,
                 mode='cat',
                 dropout=0.5, weight_decay=5e-4,
                 lr=0.005, bias=True):
        super().__init__()
        self.mode = mode
        num_JK_layers = len(list(hids)) - 1  # number of JK layers

        assert num_JK_layers >= 1 and len(set(hids)) == 1, 'the number of hidden layers should be greated than 2 and the hidden units must be equal'

        conv = []
        self.dropout = nn.Dropout(dropout)
        for hid, act in zip(hids, acts):
            conv.append(GraphConv(in_features,
                                  hid,
                                  bias=bias,
                                  activation=activations.get(act)))
            in_features = hid

        assert len(conv) == num_JK_layers + 1

        self.conv = nn.ModuleList(conv)

        if self.mode == 'cat':
            hid = hid * (num_JK_layers + 1)
        elif self.mode == 'lstm':
            self.lstm = nn.LSTM(hid, (num_JK_layers * hid) // 2, bidirectional=True, batch_first=True)
            self.attn = nn.Linear(2 * ((num_JK_layers * hid) // 2), 1)

        self.output = nn.Linear(hid, out_features)
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(self.parameters(),
                                          weight_decay=weight_decay, lr=lr),
                     metrics=[Accuracy()])

    def reset_parameters(self):
        for conv in self.conv:
            conv.reset_parameters()
        if self.mode == 'lstm':
            self.lstm.reset_parameters()
            self.attn.reset_parameters()
        self.output.reset_parameters()

    def forward(self, feats, g):
        feat_lst = []
        for conv in self.conv:
            feats = self.dropout(conv(g, feats))
            feat_lst.append(feats)

        if self.mode == 'cat':
            out = torch.cat(feat_lst, dim=-1)
        elif self.mode == 'max':
            out = torch.stack(feat_lst, dim=-1).max(dim=-1)[0]
        else:
            # lstm
            x = torch.stack(feat_lst, dim=1)
            alpha, _ = self.lstm(x)
            alpha = self.attn(alpha).squeeze(-1)
            alpha = torch.softmax(alpha, dim=-1).unsqueeze(-1)
            out = (x * alpha).sum(dim=1)

        g.ndata['h'] = out
        g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))

        return self.output(g.ndata['h'])
