import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import GCNConv, Sequential, activations
from graphgallery.nn.metrics.pytorch import Accuracy

class AGNN(TorchKeras):

    def __init__(self,
                 in_features,
                 out_features,
                 *,
                 hids=[16],
                 num_attn=2,
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=False):
        super().__init__()
        conv = []

        for hid, act in zip(hids, acts):
            conv.append(nn.Linear(in_features, hid, bias=bias))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))            
            in_features = hid
            
        # for Cora dataset, the first propagation layer is non-trainable
        # and beta is fixed at 0      
        conv.append(GraphAttentionLayer(trainable=False))
        for i in range(1, num_attn):
            conv.append(GraphAttentionLayer())
                
        conv.append(nn.Linear(in_features, out_features, bias=bias))
        conv.append(nn.Dropout(dropout))
        conv = Sequential(*conv)
        self.conv = conv
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam([dict(params=conv[0].parameters(),
                                                weight_decay=weight_decay),
                                           dict(params=conv[1:].parameters(),
                                                weight_decay=0.)], lr=lr),
                     metrics=[Accuracy()])        

    def forward(self, x, adj):
        return self.conv(x, adj)
    
class GraphAttentionLayer(nn.Module):

    def __init__(self, trainable=True):
        super(GraphAttentionLayer, self).__init__()
        
        if trainable:
            # unifrom initialization
            self.beta = nn.Parameter(torch.tensor(1.).uniform_(0, 1))
        self.trainable = trainable
        
    def forward(self, x, adj):
        # NaN grad bug fixed at pytorch 0.3. Release note:
        #     `when torch.norm returned 0.0, the gradient was NaN.
        #     We now use the subgradient at 0.0, so the gradient is 0.0.`
        norm2 = torch.norm(x, 2, 1).view(-1, 1)

        # add a minor constant (1e-7) to denominator to prevent division by
        # zero error
        if self.trainable:
            cos = self.beta * torch.div(torch.mm(x, x.t()), torch.mm(norm2, norm2.t()) + 1e-7)
        else:
            cos = 0.

        # neighborhood masking (inspired by this repo:
        # https://github.com/danielegrattarola/keras-gat)
        mask = (1. - adj) * -1e9
        masked = cos + mask

        # propagation matrix
        P = F.softmax(masked, dim=1)

        # attention-guided propagation
        output = torch.mm(P, x)
        return output
    
    def extra_repr(self):
        return f"trainable={self.trainable}"    