import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import List, Optional, Tuple

import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.utils import tqdm
from graphgallery.attack.targeted import PyTorch
from graphgallery.attack.targeted.targeted_attacker import TargetedAttacker

from collections import namedtuple
SubGraph = namedtuple('SubGraph', ['edge_index', 'non_edge_index', 'self_loop', 'self_loop_weight',
                                   'edge_weight', 'non_edge_weight', 'edges_all'])


@torch.jit.script
def maybe_dim_size(index, dim_size=None):
    # type: (Tensor, Optional[int]) -> int
    if dim_size is not None:
        return dim_size
    return int(index.max().item()) + 1 if index.numel() > 0 else 0


@torch.jit.script
def gen(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    # type: (Tensor, Tensor, int, Optional[Tensor], Optional[int], int) -> Tuple[Tensor, Tensor, Tensor, int]
    dims = torch.jit.annotate(List[int], [])
    for i in range(src.dim()):
        dims.append(i)
    dim = dims[dim]

    # Automatically expand index tensor to the right dimensions.
    if index.dim() == 1:
        index_size = torch.jit.annotate(List[int], [])
        for i in range(src.dim()):
            index_size.append(1)
        index_size[dim] = src.size(dim)
        index = index.view(index_size).expand_as(src)

    # Generate output tensor if not given.
    if out is None:
        out_size = list(src.size())
        dim_size = maybe_dim_size(index, dim_size)
        out_size[dim] = dim_size
        out = torch.empty(out_size, dtype=src.dtype, device=src.device)
        out.fill_(fill_value)

    return src, out, index, dim


@torch.jit.script
def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    # type: (Tensor, Tensor, int, Optional[Tensor], Optional[int], int) -> Tensor
    src, out, index, dim = gen(src, index, dim, out, dim_size, fill_value)
    return out.scatter_add_(dim, index, src)


@PyTorch.register()
class SGA(TargetedAttacker):
    def process(self, surrogate, reset=True):
        assert isinstance(surrogate, gg.gallery.nodeclas.SGC), surrogate

        K = surrogate.cfg.data.K  # NOTE: Be compatible with graphgallery

        W, b = surrogate.model.parameters()
        W, b = W.detach().to(self.device), b.detach().to(self.device)
        X = torch.tensor(self.graph.node_attr, device=self.device)
        self.b = b
        self.XW = F.linear(X, W)
        self.K = K
        self.logits = surrogate.predict(np.arange(self.num_nodes))
        if reset:
            self.reset()
        return self

    def reset(self):
        super().reset()
        # for the added self-loop in `gcn_norm`
        self.selfloop_degree = torch.tensor(self.degree + 1., device=self.device).float()
        self.adj_flips = {}
        self.best_wrong_label = None
        return self

    def attack(self,
               target,
               num_budgets=None,
               logit=None,
               n_influencers=3,
               direct_attack=True,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        super().attack(target, num_budgets, direct_attack, structure_attack, feature_attack)

        logit = self.logits[target]
        idx = list(set(range(logit.size)) - set([self.target_label]))
        best_wrong_label = idx[logit[idx].argmax()]
        self.best_wrong_label = torch.tensor([best_wrong_label], device=self.device).long()
        self.true_label = torch.tensor([self.target_label], device=self.device).long()
        subgraph = self.get_subgraph(n_influencers)

        # for indirect attack, the edges related to targeted node should not be considered
        if not direct_attack:
            row, col = subgraph.edge_index
            mask = torch.logical_and(row != target, col != target).float().to(self.device)
        else:
            mask = 1.0

        for it in tqdm(range(self.num_budgets), desc='Peturbing Graph', disable=disable):
            edge_grad, non_edge_grad = self.compute_gradient(subgraph)

            with torch.no_grad():
                edge_grad *= (-2 * subgraph.edge_weight + 1) * mask
                non_edge_grad *= -2 * subgraph.non_edge_weight + 1

            max_edge_grad, max_edge_idx = torch.max(edge_grad, dim=0)
            max_non_edge_grad, max_non_edge_idx = torch.max(non_edge_grad, dim=0)

            if max_edge_grad > max_non_edge_grad:
                # remove one edge
                adv_edges = subgraph.edge_index[:, max_edge_idx]
                subgraph.edge_weight.data[max_edge_idx] = 0.0
                self.selfloop_degree[adv_edges] -= 1.0
            else:
                # add one edge
                adv_edges = subgraph.non_edge_index[:, max_non_edge_idx]
                subgraph.non_edge_weight.data[max_non_edge_idx] = 1.0
                self.selfloop_degree[adv_edges] += 1.0
            u, v = adv_edges
            assert not self.is_modified(u, v)
            self.adj_flips[(u, v)] = it
        return self

    def get_subgraph(self, n_influencers=None):
        target = self.target
        best_wrong_label = self.best_wrong_label
        neighbors = self.graph.adj_matrix[target].indices
        assert neighbors.size > 0, f'target {target} is a dangling node.'
        attacker_nodes = np.where(self.graph.node_label == best_wrong_label.item())[0]
        sub_edges, sub_nodes = self.ego_subgraph()

        if self.direct_attack or n_influencers is not None:
            influencers = [target]
            attacker_nodes = np.setdiff1d(attacker_nodes, neighbors)
        else:
            influencers = neighbors

        subgraph = self.subgraph_processing(influencers, attacker_nodes, sub_nodes, sub_edges)

        if n_influencers is not None:
            if self.direct_attack:
                influencers = [target]
                attacker_nodes = self.get_topk_influencers(subgraph, k=self.num_budgets + 1)
            else:
                influencers = neighbors
                attacker_nodes = self.get_topk_influencers(subgraph, k=n_influencers)

            subgraph = self.subgraph_processing(influencers, attacker_nodes, sub_nodes, sub_edges)
        return subgraph

    def SGCCov(self, x, edge_index, edge_weight):
        row, col = edge_index
        for _ in range(self.K):
            src = x[row] * edge_weight.view(-1, 1)
            x = scatter_add(src, col, dim=-2, dim_size=x.size(0))
        return x

    def compute_gradient(self, subgraph, eps=5.0):

        edge_weight = subgraph.edge_weight
        non_edge_weight = subgraph.non_edge_weight
        self_loop_weight = subgraph.self_loop_weight
        weights = torch.cat([edge_weight, edge_weight,
                             non_edge_weight, non_edge_weight,
                             self_loop_weight
                             ], dim=0)

        weights = self.gcn_norm(subgraph.edges_all, weights, self.selfloop_degree)
        output = self.SGCCov(self.XW, subgraph.edges_all, weights)

        logit = output[self.target] + self.b
        # model calibration
        logit = F.log_softmax(logit.view(1, -1) / eps, dim=1)
        loss = F.nll_loss(logit, self.true_label) - F.nll_loss(logit, self.best_wrong_label)
        gradients = torch.autograd.grad(loss, [edge_weight, non_edge_weight], create_graph=False)
        return gradients

    def ego_subgraph(self):
        sub_edges, sub_nodes = gf.ego_graph(self.graph.adj_matrix, self.target, self.K)
        sub_edges = sub_edges.T  # shape [2, M]
        return sub_edges, sub_nodes

    def subgraph_processing(self, influencers, attacker_nodes, sub_nodes, sub_edges):
        row = np.repeat(influencers, len(attacker_nodes))
        col = np.tile(attacker_nodes, len(influencers))
        non_edges = np.row_stack([row, col])

        if len(influencers) > 1:  # indirect attack
            # TODO: considering self-loops
            mask = self.graph.adj_matrix[non_edges[0],
                                         non_edges[1]].A1 == 0
            non_edges = non_edges[:, mask]

        nodes = np.union1d(sub_nodes, attacker_nodes)
        self_loop = np.row_stack([nodes, nodes])

        edges_all = np.hstack([
            sub_edges, sub_edges[[1, 0]], non_edges,
            non_edges[[1, 0]], self_loop
        ])

        edges_all = torch.tensor(edges_all, device=self.device)
        edge_weight = nn.Parameter(torch.ones(sub_edges.shape[1], device=self.device))
        non_edge_weight = nn.Parameter(torch.zeros(non_edges.shape[1], device=self.device))
        self_loop_weight = torch.ones(nodes.shape[0], device=self.device)

        edge_index = torch.tensor(sub_edges)
        non_edge_index = torch.tensor(non_edges)
        self_loop = torch.tensor(self_loop)

        subgraph = SubGraph(edge_index=edge_index, non_edge_index=non_edge_index,
                            self_loop=self_loop, edges_all=edges_all,
                            edge_weight=edge_weight, non_edge_weight=non_edge_weight,
                            self_loop_weight=self_loop_weight)
        return subgraph

    def get_topk_influencers(self, subgraph, k):
        _, non_edge_grad = self.compute_gradient(subgraph)
        _, index = torch.topk(non_edge_grad, k=k, sorted=False)

        influencers = subgraph.non_edge_index[1][index.cpu()]
        return influencers

    @staticmethod
    def gcn_norm(edges_all, weights, degree):
        row, col = edges_all
        inv_degree = torch.pow(degree, -0.5)
        normed_weights = weights * inv_degree[row] * inv_degree[col]
        return normed_weights
