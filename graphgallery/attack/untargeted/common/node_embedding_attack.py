import numpy as np
import scipy.sparse as sp
import scipy.linalg as spl
from numba import njit

from graphgallery import functional as gf
from graphgallery.attack.utils.estimate_utils import (
    estimate_loss_with_delta_eigenvals,
    estimate_loss_with_perturbation_gradient)
from graphgallery.attack.untargeted import Common
from ..untargeted_attacker import UntargetedAttacker


@Common.register()
class NodeEmbeddingAttack(UntargetedAttacker):
    def process(self, K=100, reset=True):
        deg_matrix = sp.diags(self.degree, dtype="float64")
        adj = self.graph.adj_matrix.astype("float64")
        # generalized eigenvalues/eigenvectors
        # whether to use sparse form
        if K:
            self.vals_org, self.vecs_org = sp.linalg.eigsh(adj,
                                                           k=K,
                                                           M=deg_matrix)
        else:
            self.vals_org, self.vecs_org = spl.eigh(adj.toarray(),
                                                    deg_matrix.A)
        if reset:
            self.reset()
        return self

    def attack(self,
               num_budgets=0.05,
               dim=32,
               window_size=5,
               addition=False,
               removel=True,
               structure_attack=True,
               feature_attack=False):

        if not (addition or removel):
            raise RuntimeError(
                'Either edge addition or removel setting should be used.')

        super().attack(num_budgets, structure_attack, feature_attack)
        num_budgets = self.num_budgets

        adj = self.graph.adj_matrix

        candidates = []
        if addition:
            num_candidates = min(2 * num_budgets, self.num_edges)
            candidate = self.generate_candidates_addition(adj, num_candidates)
            candidates.append(candidate)

        if removel:
            candidate = self.generate_candidates_removal(adj)
            candidates.append(candidate)

        candidates = np.vstack(candidates)

        delta_w = 1. - 2 * adj[candidates[:, 0], candidates[:, 1]].A1

        loss_for_candidates = estimate_loss_with_delta_eigenvals(
            candidates, delta_w, self.vals_org, self.vecs_org, self.num_nodes,
            dim, window_size)

        self.dim = dim
        self.adj_flips = candidates[loss_for_candidates.argsort()
                                    [-num_budgets:]]
        self.window_size = window_size

    def generate_candidates_removal(self, adj):
        """Generates candidate edge flips for removal (edge -> non-edge),
         disallowing one random edge per node to prevent singleton nodes.

        :return: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        """
        num_nodes = self.num_nodes
        deg = np.where(self.degree == 1)[0]

        hiddeen = np.column_stack((np.arange(num_nodes),
                                   np.fromiter(map(np.random.choice,
                                                   adj.tolil().rows),
                                               dtype=np.int32)))

        adj_hidden = gf.edge_to_sparse_adj(hiddeen, shape=adj.shape)
        adj_hidden = adj_hidden.maximum(adj_hidden.T)
        adj_keep = adj - adj_hidden
        candidates = np.transpose(sp.triu(adj_keep).nonzero())

        candidates = candidates[np.logical_not(
            np.in1d(candidates[:, 0], deg) | np.in1d(candidates[:, 1], deg))]

        return candidates

    def generate_candidates_addition(self, adj, num_candidates):
        """Generates candidate edge flips for addition (non-edge -> edge).

        :param num_candidates: int
            Number of candidates to generate.
        :return: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        """
        num_nodes = self.num_nodes

        candidates = np.random.randint(0, num_nodes, [num_candidates * 5, 2])
        candidates = candidates[candidates[:, 0] < candidates[:, 1]]
        candidates = candidates[adj[candidates[:, 0], candidates[:,
                                                                 1]].A1 == 0]
        candidates = np.array(list(set(map(tuple, candidates))))
        candidates = candidates[:num_candidates]

        return candidates
