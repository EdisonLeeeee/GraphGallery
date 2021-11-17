import torch.nn as nn
from graphgallery.nn.layers.pytorch import WaveletConv, Sequential, activations


class GWNN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 num_nodes,
                 *,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 bias=False):
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

    def forward(self, x, wavelet, inverse_wavelet):
        return self.conv(x, wavelet, inverse_wavelet)
