import numpy as np
import scipy.sparse as sp
from graphgallery import functional as gf
from graphgallery.utils import tqdm
from graphgallery.attack.untargeted import Common
from ..untargeted_attacker import UntargetedAttacker


@Common.register()
class Degree(UntargetedAttacker):
    """For each perturbation, inserting or removing an edge based on degree centrality, which is equivalent to the sum of degrees in original graph
    """

    def process(self, reset=True):
        self.nodes_set = set(range(self.num_nodes))
        if reset:
            self.reset()
        return self

    def attack(self,
               num_budgets=0.05,
               complement=False,
               addition=True,
               removel=False,
               structure_attack=True,
               feature_attack=False):

        super().attack(num_budgets, structure_attack, feature_attack)
        num_budgets = self.num_budgets

        candidates = []
        if addition:
            num_candidates = min(2 * num_budgets, self.num_edges)
            candidates.append(
                self.generate_candidates_addition(num_candidates))

        if removel:
            candidates.append(self.generate_candidates_removal())

        candidates = np.vstack(candidates)

        deg_argsort = (self.degree[candidates[:, 0]] +
                       self.degree[candidates[:, 1]]).argsort()
        self.adj_flips = candidates[deg_argsort[-num_budgets:]]

    def generate_candidates_removal(self):
        """Generates candidate edge flips for removal (edge -> non-edge),
         disallowing one random edge per node to prevent singleton nodes.

        :return: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        """
        num_nodes = self.num_nodes
        deg = np.where(self.degree == 1)[0]
        adj = self.graph.adj_matrix

        hiddeen = np.column_stack((np.arange(num_nodes),
                                   np.fromiter(map(np.random.choice,
                                                   adj.tolil().rows),
                                               dtype=np.int32)))

        adj_hidden = gf.edge_to_sparse_adj(hiddeen, shape=adj.shape)
        adj_hidden = adj_hidden.maximum(adj_hidden.T)
        adj_keep = adj - adj_hidden
        candidates = np.transpose(sp.triu(adj_keep, k=-1).nonzero())

        candidates = candidates[np.logical_not(
            np.in1d(candidates[:, 0], deg) | np.in1d(candidates[:, 1], deg))]

        return candidates

    def generate_candidates_addition(self, num_candidates):
        """Generates candidate edge flips for addition (non-edge -> edge).

        :param num_candidates: int
            Number of candidates to generate.
        :return: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        """
        num_nodes = self.num_nodes
        adj = self.graph.adj_matrix

        candidates = np.random.randint(0, num_nodes, [num_candidates * 5, 2])
        candidates = candidates[candidates[:, 0] < candidates[:, 1]]
        candidates = candidates[adj[candidates[:, 0], candidates[:,
                                                                 1]].A1 == 0]
        candidates = np.array(list(set(map(tuple, candidates))))
        candidates = candidates[:num_candidates]

        return candidates
