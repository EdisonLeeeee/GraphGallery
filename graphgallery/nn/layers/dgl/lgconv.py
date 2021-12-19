import math
import torch
from torch import nn
from dgl import function as fn
from scipy.special import factorial


class LGConv(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 k=1,
                 cached=False,
                 bias=True):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self._cached = cached
        self._cached_h = None
        self.k = k
        self.alpha = nn.ParameterList()
        for _ in range(k + 1):
            self.alpha.append(nn.Parameter(torch.Tensor(1)))        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.fc.weight.data.uniform_(-stdv, stdv)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

        stdvk = 1. / math.sqrt(self.k)
        for i in range(self.k + 1):
            self.alpha[i].data.uniform_(-stdvk, stdvk)        

    def forward(self, graph, feat):
        with graph.local_scope():
            if self._cached_h is not None:
                feat_list = self._cached_h
                result = 0.
                for i, k_feat in enumerate(feat_list):
                    result += self.fc(k_feat * self.alpha[i])
            else:
                feat_list = []

                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)

                feat_list.append(feat.float())

                for i in range(self.k):
                    feat = feat * norm
                    feat = feat.float()
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'),
                                     fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm
                    feat_list.append(feat)

                result = 0.
                for i, k_feat in enumerate(feat_list):
                    result += self.fc(k_feat * self.alpha[i])
                    
                # cache feature
                if self._cached:
                    self._cached_h = feat_list
            return result

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, k={self.k})"

class EGConv(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 k=1,
                 cached=False,
                 bias=True):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self._cached = cached
        self._cached_h = None
        self.k = k
        self._lambda = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.fc.weight.data.uniform_(-stdv, stdv)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

        stdvk = 1. / math.sqrt(self.k)
        self._lambda.data.uniform_(-stdvk, stdvk)


    def forward(self, graph, feat):
        with graph.local_scope():
            if self._cached_h is not None:
                feat_list = self._cached_h
                result = 0.
                for i, k_feat in enumerate(feat_list):
                    result += self.fc(k_feat * (self._lambda.pow(i) / factorial(i)))
            else:
                feat_list = []

                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)

                feat_list.append(feat.float())

                for i in range(self.k):
                    feat = feat * norm
                    feat = feat.float()
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'),
                                     fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm
                    feat_list.append(feat)

                result = 0.
                for i, k_feat in enumerate(feat_list):
                    result += self.fc(k_feat*(self._lambda.pow(i)/factorial(i)))
                    
                # cache feature
                if self._cached:
                    self._cached_h = feat_list
            return result

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, k={self.k})"
    

class hLGConv(LGConv):
    def __init__(self,
                 in_features,
                 out_features,
                 k=1,
                 cached=False,
                 bias=True):

        super().__init__(in_features, out_features, k=k, 
                         cached=cached, bias=bias)    
        self.lambada_fun = nn.Linear(in_features + out_features, in_features)
        
    def forward(self, graph, feat):
        with graph.local_scope():
            if self._cached_h is not None:
                feat_list = self._cached_h
                feat_km1 = feat_list[0]
                result = feat_km1.new_zeros(feat_km1.size(0), self.out_features)
                for i, k_feat in enumerate(feat_list):
                    _dLambda = self.lambada_fun(torch.cat([feat_km1, result], dim=1))
                    result += self.fc(k_feat * _dLambda)
                    feat_km1 = k_feat
            else:
                feat_list = []

                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)

                feat_list.append(feat.float())

                for i in range(self.k):
                    feat = feat * norm
                    feat = feat.float()
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'),
                                     fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm
                    feat_list.append(feat)

                feat_km1 = feat_list[0]
                result = feat_km1.new_zeros(feat_km1.size(0), self.out_features)
                for i, k_feat in enumerate(feat_list):
                    _dLambda = self.lambada_fun(torch.cat([feat_km1, result], dim=1))
                    result += self.fc(k_feat * _dLambda)
                    feat_km1 = k_feat
                    
                # cache feature
                if self._cached:
                    self._cached_h = feat_list
            return result        