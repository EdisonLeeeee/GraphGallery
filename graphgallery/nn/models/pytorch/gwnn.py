import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchEngine
from graphgallery.nn.layers.pytorch import WaveletConv, Sequential, activations
from graphgallery.nn.metrics.pytorch import Accuracy


class GWNN(TorchEngine):
    def __init__(self,
                 in_features,
                 out_features,
                 num_nodes,
                 *,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01, bias=False):
        super().__init__()
        conv = []
        conv.append(nn.Dropout(dropout))
        for hid, act in zip(hids, acts):
            conv.append(WaveletConv(in_features,
                                    hid, num_nodes=num_nodes,
                                    bias=bias))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid
        conv.append(WaveletConv(in_features, out_features, num_nodes=num_nodes, bias=bias))
        conv = Sequential(*conv)

        self.conv = conv
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(self.parameters(),
                                          weight_decay=weight_decay, lr=lr),
                     metrics=[Accuracy()])

    def forward(self, x, wavelet, inverse_wavelet):
        return self.conv(x, wavelet, inverse_wavelet)
