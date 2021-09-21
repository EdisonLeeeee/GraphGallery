# ==============================================================================
# Most of the codes come from DeepRobust: https://github.com/DSE-MSU/DeepRobust
# Copyright The DeepRobust Authors. All Rights Reserved.
# Licensed under the MIT License
# https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/global_attack/mettack.py
# ==============================================================================

import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter

import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.utils import tqdm
from graphgallery.nn.init.pytorch import glorot_uniform, zeros
# from graphadv.utils.graph_utils import likelihood_ratio_filter
from graphgallery.attack.untargeted import PyTorch
from ..untargeted_attacker import UntargetedAttacker


class BaseMeta(UntargetedAttacker):
    """Base model for Mettack."""
    # mettack can also conduct feature attack
    _allow_feature_attack = True

    def process(self,
                train_nodes,
                unlabeled_nodes,
                self_training_labels,
                hids,
                use_relu,
                reset=True):
        self.ll_ratio = None

        self.train_nodes = gf.astensor(train_nodes,
                                       dtype=self.intx,
                                       device=self.device)
        self.unlabeled_nodes = gf.astensor(unlabeled_nodes,
                                           dtype=self.intx,
                                           device=self.device)
        self.labels_train = gf.astensor(self.graph.node_label[train_nodes],
                                        dtype=self.intx,
                                        device=self.device)
        self.self_training_labels = gf.astensor(self_training_labels,
                                                dtype=self.intx,
                                                device=self.device)
        self.adj_tensor = gf.astensor(self.graph.adj_matrix.A,
                                      dtype=self.floatx,
                                      device=self.device)
        self.x_tensor = gf.astensor(self.graph.node_attr,
                                    dtype=self.floatx,
                                    device=self.device)
        self.build(hids=hids)
        self.use_relu = use_relu

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.adj_changes = None
        self.x_changes = None

        if reset:
            self.reset()
        return self

    def reset(self):
        super().reset()
        self.adj_flips = []
        self.nattr_flips = []
        self.adj_changes = Parameter(torch.zeros_like(self.adj_tensor)).to(self.device)
        self.x_changes = Parameter(torch.zeros_like(self.x_tensor)).to(self.device)
        return self

    def get_perturbed_adj(self, adj, adj_changes):
        adj_changes_square = adj_changes - torch.diag(torch.diag(adj_changes, 0))
        adj_changes_symm = self.clip(adj_changes_square + torch.transpose(adj_changes_square, 1, 0))
        modified_adj = adj_changes_symm + adj
        return modified_adj

    def get_perturbed_x(self, x, x_changes):
        return x + self.clip(x_changes)

    def forward(self, x, adj):
        h = x
        for w in self.weights[:-1]:
            h = adj @ h @ w
            if self.use_relu:
                h = F.relu(h)

        return adj @ h @ self.weights[-1]

    def filter_potential_singletons(self, modified_adj):
        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        logical_and_symmetric = l_and + l_and.t()
        flat_mask = 1 - logical_and_symmetric
        return flat_mask

    # def log_likelihood_constraint(self, modified_adj, adj, ll_cutoff):
    #     """
    #     Computes a mask for entries that, if the edge corresponding to the entry is added/removed, would lead to the
    #     log likelihood constraint to be violated.

    #     Note that different data type (float, double) can effect the final results.
    #     """
    #     t_d_min = torch.tensor(2.0).to(self.device)
    #     t_possible_edges = np.array(
    #         np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()).T
    #     allowed_mask, current_ratio = utils.likelihood_ratio_filter(
    #         t_possible_edges, modified_adj, adj, t_d_min, ll_cutoff)
    #     return allowed_mask, current_ratio

    def structure_score(self,
                        modified_adj,
                        adj_grad,
                        ll_constraint=None,
                        ll_cutoff=None):

        adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
        # Make sure that the minimum entry is 0.
        adj_meta_grad -= adj_meta_grad.min()
        # Filter self-loops
        adj_meta_grad -= torch.diag(torch.diag(adj_meta_grad, 0))
        # Set entries to 0 that could lead to singleton nodes.
        singleton_mask = self.filter_potential_singletons(modified_adj)
        adj_meta_grad = adj_meta_grad * singleton_mask

        # if ll_constraint:
        #     allowed_mask, self.ll_ratio = self.log_likelihood_constraint(
        #         modified_adj, adj, ll_cutoff)
        #     allowed_mask = allowed_mask.to(self.device)
        #     adj_meta_grad = adj_meta_grad * allowed_mask
        return adj_meta_grad.flatten()

    def feature_score(self, modified_nx, x_grad):
        feature_meta_grad = x_grad * (-2 * modified_nx + 1)
        feature_meta_grad -= feature_meta_grad.min()
        return feature_meta_grad.flatten()

    def clip(self, matrix):
        clipped_matrix = torch.clamp(matrix, -1., 1.)
        return clipped_matrix


@PyTorch.register()
class Metattack(BaseMeta):
    def process(self,
                train_nodes,
                unlabeled_nodes,
                self_training_labels,
                hids=[16],
                lr=0.1,
                epochs=100,
                momentum=0.9,
                lambda_=0.,
                use_relu=True,
                reset=True):

        self.lr = lr
        self.epochs = epochs
        self.momentum = momentum
        self.lambda_ = lambda_

        if lambda_ not in (0., 0.5, 1.):
            raise ValueError(
                'Invalid value of `lambda_`, allowed values [0: (meta-self), 1: (meta-train), 0.5: (meta-both)].'
            )
        return super().process(train_nodes=train_nodes,
                               unlabeled_nodes=unlabeled_nodes,
                               self_training_labels=self_training_labels,
                               hids=hids,
                               use_relu=use_relu,
                               reset=reset)

    def build(self, hids):
        hids = gf.repeat(hids)
        weights, w_velocities = [], []

        pre_hid = self.num_attrs
        for hid in hids + [self.num_classes]:
            shape = (pre_hid, hid)
            w = Parameter(torch.zeros(shape).to(self.device))
            w_velocity = torch.zeros(shape).to(self.device)
            weights.append(w)
            w_velocities.append(w_velocity)

            pre_hid = hid

        self.weights, self.w_velocities = weights, w_velocities
        self._initialize()

    def _initialize(self):
        for w, wv in zip(self.weights, self.w_velocities):
            glorot_uniform(w)
            zeros(wv)

        for ix in range(len(self.weights)):
            self.weights[ix] = self.weights[ix].detach()
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.weights[ix].requires_grad = True

    def inner_train(self, x, adj):
        self._initialize()
        train_nodes = self.train_nodes
        train_labels = self.labels_train

        for it in range(self.epochs):
            output = self.forward(x, adj)
            loss = self.loss_fn(output[train_nodes], train_labels)

            weight_grads = torch.autograd.grad(loss,
                                               self.weights,
                                               create_graph=True)

            self.w_velocities = [
                self.momentum * v + g
                for v, g in zip(self.w_velocities, weight_grads)
            ]

            self.weights = [
                w - self.lr * v for w, v in zip(self.weights, self.w_velocities)
            ]

    def meta_grad(self, x, adj, calibration=1.0):

        output = self.forward(x, adj) / calibration
        loss_labeled = self.loss_fn(output[self.train_nodes], self.labels_train)
        loss_unlabeled = self.loss_fn(output[self.unlabeled_nodes], self.self_training_labels)
        attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        x_grad = adj_grad = None

        if self.feature_attack:
            retain_graph = True if self.structure_attack else False
            x_grad = torch.autograd.grad(attack_loss, self.x_changes, retain_graph=retain_graph)[0]

        if self.structure_attack:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes)[0]

        return x_grad, adj_grad

    def attack(self,
               num_budgets=0.05,
               structure_attack=True,
               feature_attack=False,
               ll_constraint=False,
               ll_cutoff=0.004,
               disable=False):

        super().attack(num_budgets, structure_attack, feature_attack)

        if ll_constraint:
            raise NotImplementedError(
                "`log_likelihood_constraint` has not been well tested."
                " Please set `ll_constraint=False` to achieve a better performance."
            )

        if feature_attack and not self.graph.is_binary():
            raise ValueError(
                "Attacks on the node features are currently only supported for binary attributes."
            )

        modified_adj, modified_nx = self.adj_tensor, self.x_tensor
        adj_tensor, x_tensor = self.adj_tensor, self.x_tensor
        adj_changes, x_changes = self.adj_changes, self.x_changes
        adj_flips, nattr_flips = self.adj_flips, self.nattr_flips

        self.inner_train(modified_nx, modified_adj)

        for it in tqdm(range(self.num_budgets),
                       desc='Peturbing Graph',
                       disable=disable):

            if structure_attack:
                modified_adj = self.get_perturbed_adj(adj_tensor, adj_changes)

            if feature_attack:
                modified_nx = self.get_perturbed_x(x_tensor, x_changes)

            adj_norm = gf.normalize_adj_tensor(modified_adj)

            self.inner_train(modified_nx, adj_norm)

            x_grad, adj_grad = self.meta_grad(modified_nx, adj_norm)

            x_meta_score = torch.tensor(0.0)
            adj_meta_score = torch.tensor(0.0)

            if structure_attack:
                adj_meta_score = self.structure_score(modified_adj, adj_grad,
                                                      ll_constraint, ll_cutoff)
            if feature_attack:
                x_meta_score = self.feature_score(modified_nx, x_grad)

            if adj_meta_score.max() >= x_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)
                row, col = unravel_index(adj_meta_argmax, self.num_nodes)
                self.adj_changes.data[row][col] += -2 * modified_adj[row][col] + 1
                self.adj_changes.data[col][row] += -2 * modified_adj[row][col] + 1
                adj_flips.append((row, col))

            else:
                x_meta_argmax = torch.argmax(x_meta_score)
                row, col = unravel_index(x_meta_argmax, self.num_attrs)
                self.x_changes.data[row][col] += -2 * modified_nx[row][col] + 1
                nattr_flips.append((row, col))


def unravel_index(index, shape):
    row = index // shape
    col = index % shape
    return row, col
