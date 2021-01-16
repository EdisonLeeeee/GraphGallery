import numpy as np
from numba import njit

import scipy.sparse as sp
import scipy.linalg as spl

import tensorflow as tf
from graphgallery import functional as gf
from graphgallery.attack.targeted import Common
from graphgallery.attack.utils.estimate_utils import (
    estimate_loss_with_delta_eigenvals,
    estimate_loss_with_perturbation_gradient)
from ..targeted_attacker import TargetedAttacker


@Common.register()
class NodeEmbeddingAttack(TargetedAttacker):
    def process(self, K=50, reset=True):
        self.nodes_set = set(range(self.num_nodes))

        deg_matrix = sp.diags(self.degree, dtype="float64")
        self.vals_org, self.vecs_org = sp.linalg.eigsh(
            self.graph.adj_matrix.astype('float64'), k=K, M=deg_matrix)
        if reset:
            self.reset()
        return self

    def attack(self,
               target,
               num_budgets=None,
               dim=32,
               window_size=5,
               n_neg_samples=3,
               direct_attack=True,
               structure_attack=True,
               feature_attack=False):

        super().attack(target, num_budgets, direct_attack, structure_attack,
                       feature_attack)
        num_budgets = self.num_budgets
        num_nodes = self.num_nodes
        adj = self.graph.adj_matrix

        if direct_attack:
            influence_nodes = [target]
            candidates = np.column_stack(
                (np.tile(target,
                         num_nodes - 1), list(self.nodes_set - set([target]))))
        else:
            influence_nodes = adj[target].indices
            candidates = np.row_stack([
                np.column_stack((np.tile(infl, num_nodes - 2),
                                 list(self.nodes_set - set([target, infl]))))
                for infl in influence_nodes
            ])
        if not self.allow_singleton:
            candidates = gf.singleton_filter(candidates, adj)

        delta_w = 1. - 2 * adj[candidates[:, 0], candidates[:, 1]].A1
        loss_for_candidates = estimate_loss_with_delta_eigenvals(
            candidates, delta_w, self.vals_org, self.vecs_org, self.num_nodes,
            dim, window_size)

        self.adj_flips = candidates[loss_for_candidates.argsort()
                                    [-num_budgets:]]
        return self
