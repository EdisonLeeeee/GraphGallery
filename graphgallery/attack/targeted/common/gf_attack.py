import numpy as np
import scipy.sparse as sp
from scipy import linalg

from graphgallery.utils import tqdm
from graphgallery import functional as gf
from graphgallery.attack.targeted import Common
from ..targeted_attacker import TargetedAttacker
from .nettack import compute_alpha, update_Sx, compute_log_likelihood, filter_chisquare


@Common.register()
class GFA(TargetedAttacker):
    """
    T=128 for citeseer and pubmed, 
    T=_N//2 for cora to reproduce results in paper.
    """

    def process(self, K=2, T=128, reset=True):
        adj, x = self.graph.adj_matrix, self.graph.node_attr

        adj_with_I = adj + sp.eye(adj.shape[0])
        rowsum = adj_with_I.sum(1).A1
        degree_mat = np.diag(rowsum)
        eig_vals, eig_vec = linalg.eigh(adj_with_I.A, degree_mat)
        X_mean = np.sum(x, axis=1)

        # The order of graph filter K
        self.K = K

        # Top-T largest eigen-values/vectors selected
        self.T = T

        self.eig_vals, self.eig_vec = eig_vals, eig_vec
        self.X_mean = X_mean
        if reset:
            self.reset()
        return self

    def reset(self):
        super().reset()
        self.modified_adj = self.graph.adj_matrix.tolil(copy=True)
        return self

    def attack(self,
               target,
               num_budgets=None,
               direct_attack=True,
               structure_attack=True,
               feature_attack=False,
               ll_constraint=False,
               ll_cutoff=0.004,
               disable=False):

        super().attack(target, num_budgets, direct_attack, structure_attack,
                       feature_attack)

        # Setup starting values of the likelihood ratio test.
        degree_sequence_start = self.degree
        current_degree_sequence = self.degree.astype('float64')
        d_min = 2  # denotes the minimum degree a node needs to have to be considered in the power-law test
        S_d_start = np.sum(
            np.log(degree_sequence_start[degree_sequence_start >= d_min]))
        current_S_d = np.sum(
            np.log(current_degree_sequence[current_degree_sequence >= d_min]))
        n_start = np.sum(degree_sequence_start >= d_min)
        current_n = np.sum(current_degree_sequence >= d_min)
        alpha_start = compute_alpha(n_start, S_d_start, d_min)
        log_likelihood_orig = compute_log_likelihood(n_start, alpha_start,
                                                     S_d_start, d_min)

        N = self.num_nodes
        if not direct_attack:
            # Choose influencer nodes
            # influence_nodes = self.graph.adj_matrix[target].nonzero()[1]
            influence_nodes = self.graph.adj_matrix[target].indices
            # Potential edges are all edges from any attacker to any other node, except the respective
            # attacker itself or the node being attacked.
            potential_edges = np.row_stack([
                np.column_stack((np.tile(infl, N - 2),
                                 np.setdiff1d(np.arange(N),
                                              np.array([target, infl]))))
                for infl in influence_nodes
            ])
        else:
            # direct attack
            potential_edges = np.column_stack(
                (np.tile(target, N - 1), np.setdiff1d(np.arange(N), target)))
            influence_nodes = np.asarray([target])

        for it in tqdm(range(self.num_budgets),
                       desc='Peturbing Graph',
                       disable=disable):

            if not self.allow_singleton:
                filtered_edges = gf.singleton_filter(potential_edges,
                                                     self.modified_adj)
            else:
                filtered_edges = potential_edges

            if ll_constraint:
                # Update the values for the power law likelihood ratio test.
                deltas = 2 * (1 - self.modified_adj[tuple(
                    filtered_edges.T)].toarray()[0]) - 1
                d_edges_old = current_degree_sequence[filtered_edges]
                d_edges_new = current_degree_sequence[
                    filtered_edges] + deltas[:, None]
                new_S_d, new_n = update_Sx(current_S_d, current_n, d_edges_old,
                                           d_edges_new, d_min)
                new_alphas = compute_alpha(new_n, new_S_d, d_min)
                new_ll = compute_log_likelihood(new_n, new_alphas, new_S_d,
                                                d_min)
                alphas_combined = compute_alpha(new_n + n_start,
                                                new_S_d + S_d_start, d_min)
                new_ll_combined = compute_log_likelihood(
                    new_n + n_start, alphas_combined, new_S_d + S_d_start,
                    d_min)
                new_ratios = -2 * new_ll_combined + 2 * (new_ll +
                                                         log_likelihood_orig)

                # Do not consider edges that, if added/removed, would lead to a violation of the
                # likelihood ration Chi_square cutoff value.
                powerlaw_filter = filter_chisquare(new_ratios, ll_cutoff)
                filtered_edges = filtered_edges[powerlaw_filter]

            struct_scores = self.struct_score(self.modified_adj,
                                              self.X_mean,
                                              self.eig_vals,
                                              self.eig_vec,
                                              filtered_edges,
                                              K=self.K,
                                              T=self.T,
                                              lambda_method="nosum")
            best_edge_ix = struct_scores.argmax()
            u, v = filtered_edges[best_edge_ix]  # best edge

            while (u, v) in self.adj_flips:
                struct_scores[best_edge_ix] = 0
                best_edge_ix = struct_scores.argmax()
                u, v = filtered_edges[best_edge_ix]

            self.modified_adj[(u, v)] = self.modified_adj[(
                v, u)] = 1. - self.modified_adj[(u, v)]
            self.adj_flips[(u, v)] = 1.0

            if ll_constraint:
                # Update likelihood ratio test values
                current_S_d = new_S_d[powerlaw_filter][best_edge_ix]
                current_n = new_n[powerlaw_filter][best_edge_ix]
                current_degree_sequence[[
                    u, v
                ]] += deltas[powerlaw_filter][best_edge_ix]
        return self

    @staticmethod
    def struct_score(A,
                     X_mean,
                     eig_vals,
                     eig_vec,
                     filtered_edges,
                     K,
                     T,
                     lambda_method="nosum"):
        '''
        Calculate the scores as formulated in paper.

        Parameters
        ----------
        K: int, default: 2
            The order of graph filter K.

        T: int, default: 128
            Selecting the Top-T largest eigen-values/vectors.

        lambda_method: "sum"/"nosum", default: "nosum"
            Indicates the scores are calculated from which loss as in Equation (8) or Equation (12).
            "nosum" denotes Equation (8), where the loss is derived from Graph Convolutional Networks,
            "sum" denotes Equation (12), where the loss is derived from Sampling-based Graph Embedding Methods.

        Returns
        -------
        Scores for candidate edges.

        '''
        results = []
        A = A + sp.eye(A.shape[0])
        #         A[A > 1] = 1
        rowsum = A.sum(1).A1
        D_min = rowsum.min()
        abs_V = len(eig_vals)
        return_values = []

        for j, (u, v) in enumerate(filtered_edges):
            # eig_vals_res = np.zeros(len(eig_vals))
            eig_vals_res = (1 - 2 * A[(u, v)]) * (
                2 * eig_vec[u, :] * eig_vec[v, :] - eig_vals *
                (np.square(eig_vec[u, :]) + np.square(eig_vec[v, :])))
            eig_vals_res = eig_vals + eig_vals_res

            if lambda_method == "sum":
                if K == 1:
                    eig_vals_res = np.abs(eig_vals_res / K) * (1 / D_min)
                else:
                    for itr in range(1, K):
                        eig_vals_res = eig_vals_res + np.power(
                            eig_vals_res, itr + 1)
                    eig_vals_res = np.abs(eig_vals_res / K) * (1 / D_min)
            else:
                eig_vals_res = np.square(
                    (eig_vals_res + np.ones(len(eig_vals_res))))
                eig_vals_res = np.power(eig_vals_res, K)

            eig_vals_idx = np.argsort(eig_vals_res)  # from small to large
            eig_vals_k_sum = eig_vals_res[eig_vals_idx[:T]].sum()
            u_k = eig_vec[:, eig_vals_idx[:T]]
            u_x_mean = u_k.T.dot(X_mean)
            return_values.append(eig_vals_k_sum *
                                 np.square(np.linalg.norm(u_x_mean)))

        return np.asarray(return_values)
