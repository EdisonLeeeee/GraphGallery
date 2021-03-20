import numpy as np
import scipy.sparse as sp
import scipy.linalg as spl

from graphgallery import functional as gf
from graphgallery.attack.utils.estimate_utils import (
    estimate_loss_with_delta_eigenvals,
    estimate_loss_with_perturbation_gradient)
from graphgallery.attack.untargeted import Common
from ..untargeted_attacker import UntargetedAttacker


@Common.register()
class NodeEmbeddingAttack(UntargetedAttacker):

    def attack(self,
               num_budgets=0.05,
               dim=32,
               window_size=5,
               K=100,
               attack_type="add_by_remove",
               structure_attack=True,
               feature_attack=False):

        if not attack_type in ["remove", "add", "add_by_remove"]:
            raise RuntimeError(
                'attack_type should be one of "remove", "add", "add_by_remove".')

        super().attack(num_budgets, structure_attack, feature_attack)
        num_budgets = self.num_budgets

        adj = self.graph.adj_matrix

        if attack_type.startswith("add"):
            num_candidates = min(5 * num_budgets, self.num_edges)
            # num_candidates = 10000
            candidates = self.generate_candidates_addition(adj, num_candidates)
        else:
            candidates = self.generate_candidates_removal(adj)

        if attack_type == "add_by_remove":
            adj = gf.flip_adj(adj, candidates)
            deg_matrix = sp.diags(adj.sum(1).A1, dtype=adj.dtype)
            if K:
                vals_org, vecs_org = sp.linalg.eigsh(adj, k=K, M=deg_matrix)
            else:
                vals_org, vecs_org = spl.eigh(adj.toarray(), deg_matrix.toarray())
            delta_w = 1. - 2 * adj[candidates[:, 0], candidates[:, 1]].A1

            loss_for_candidates = estimate_loss_with_delta_eigenvals(candidates, delta_w,
                                                                     vals_org, vecs_org, self.num_nodes, dim, window_size)

            self.adj_flips = candidates[loss_for_candidates.argsort()
                                        [:num_budgets]]
        else:
            # vector indicating whether we are adding an edge (+1) or removing an edge (-1)
            delta_w = 1. - 2 * adj[candidates[:, 0], candidates[:, 1]].A1

            deg_matrix = sp.diags(adj.sum(1).A1, dtype=adj.dtype)
            if K:
                vals_org, vecs_org = sp.linalg.eigsh(adj, k=K, M=deg_matrix)
            else:
                vals_org, vecs_org = spl.eigh(adj.toarray(), deg_matrix.toarray())

            loss_for_candidates = estimate_loss_with_delta_eigenvals(candidates, delta_w,
                                                                     vals_org, vecs_org, self.num_nodes, dim, window_size)
            self.adj_flips = candidates[loss_for_candidates.argsort()[-num_budgets:]]
        return self

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
        assert len(candidates) == num_candidates
        return candidates
