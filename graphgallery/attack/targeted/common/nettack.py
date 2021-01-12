"""
Implementation of the method proposed in the paper:
'Adversarial Attacks on Neural Networks for Graph Data'
by Daniel Zügner, Amir Akbarnejad and Stephan Günnemann,
published at SIGKDD'18, August 2018, London, UK

Copyright (C) 2018
Daniel Zügner
Technical University of Munich
"""
import warnings
import numpy as np
import scipy.sparse as sp
from numba import njit

import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.attack.targeted import Common
from graphgallery.utils import tqdm
from ..targeted_attacker import TargetedAttacker


@Common.register()
class Nettack(TargetedAttacker):
    """
    Nettack class used for poisoning attacks on node classification models.
    Copyright (C) 2018
    Daniel Zügner
    Technical University of Munich
    """
    # nettack can conduct feature attack
    _allow_feature_attack = True

    def process(self, W_surrogate, reset=True):

        self.W = W_surrogate
        sparse_x = sp.csr_matrix(self.graph.node_attr)
        self.cooc_matrix = sparse_x.T @ sparse_x
        self.sparse_x = sparse_x
        if reset:
            self.reset()
        return self

    def reset(self):
        super().reset()
        self.modified_adj = self.graph.adj_matrix.copy()
        self.modified_nx = self.sparse_x.copy()
        self.adj_norm = gf.normalize_adj(self.modified_adj)

        self.adj_flips = []
        self.nattr_flips = []
        self.influence_nodes = []
        self.potential_edges = []
        self.cooc_constraint = None
        return self

    def compute_cooccurrence_constraint(self, nodes):
        """
        Co-occurrence constraint as described in the paper.

        Parameters
        ----------
        nodes: np.array
            Nodes whose features are considered for change

        Returns
        -------
        np.array [len(nodes), num_attrs], dtype bool
            Binary matrix of dimension len(nodes) x num_attrs. A 1 in entry n,d indicates that
            we are allowed to add feature d to the features of node n.

        """

        num_nodes, num_attrs = self.modified_nx.shape
        words_graph = self.cooc_matrix - sp.diags(self.cooc_matrix.diagonal())
        words_graph.eliminate_zeros()
        #         words_graph.setdiag(0)
        words_graph.data = words_graph.data > 0
        word_degrees = words_graph.sum(0).A1

        inv_word_degrees = np.reciprocal(word_degrees.astype(float) + 1e-8)

        sd = np.zeros(num_nodes)
        for n in range(num_nodes):
            n_idx = self.modified_nx[n, :].nonzero()[1]
            sd[n] = np.sum(inv_word_degrees[n_idx.tolist()])

        scores_matrix = sp.lil_matrix((num_nodes, num_attrs))

        for n in nodes:
            common_words = words_graph.multiply(self.modified_nx[n])
            idegs = inv_word_degrees[common_words.nonzero()[1]]
            nnz = common_words.nonzero()[0]
            scores = np.array(
                [idegs[nnz == ix].sum() for ix in range(num_attrs)])
            scores_matrix[n] = scores
        self.cooc_constraint = sp.csr_matrix(
            scores_matrix - 0.5 * sd[:, None] > 0)

    def gradient_wrt_x(self, label):
        """
        Compute the gradient of the logit belonging to the class of the input label with respect to the input features.

        Parameters
        ----------
        label: int
            Class whose logits are of interest

        Returns
        -------
        np.array [num_nodes, num_attrs] matrix containing the gradients.

        """

        return (self.adj_norm @ self.adj_norm)[self.target].T @ sp.coo_matrix(
            self.W[:, label].reshape(1, -1))

    def compute_logits(self):
        """
        Compute the logits of the surrogate model, i.e. linearized GCN.

        Returns
        -------
        np.array, [num_nodes, num_classes]
            The log probabilities for each node.

        """
        return (self.adj_norm @ self.adj_norm @ self.modified_nx
                @ self.W)[self.target].ravel()

    def strongest_wrong_class(self, logits):
        """
        Determine the incorrect class with largest logits.

        Parameters
        ----------
        logits: np.array, [num_nodes, num_classes]
            The input logits

        Returns
        -------
        np.array, [num_nodes, L]
            The indices of the wrong labels with the highest attached log probabilities.
        """

        target_label_onehot = np.eye(self.num_classes)[self.target_label]
        return (logits - 1000 * target_label_onehot).argmax()

    def feature_scores(self):
        """
        Compute feature scores for all possible feature changes.
        """

        if self.cooc_constraint is None:
            self.compute_cooccurrence_constraint(self.influence_nodes)

        logits = self.compute_logits()
        best_wrong_class = self.strongest_wrong_class(logits)
        gradient = self.gradient_wrt_x(
            self.target_label) - self.gradient_wrt_x(best_wrong_class)
        surrogate_loss = logits[self.target_label] - logits[best_wrong_class]

        gradients_flipped = (gradient * -1).tolil()
        gradients_flipped[self.modified_nx.nonzero()] *= -1

        X_influencers = sp.lil_matrix(self.modified_nx.shape)
        X_influencers[self.influence_nodes] = self.modified_nx[
            self.influence_nodes]
        gradients_flipped = gradients_flipped.multiply(
            (self.cooc_constraint + X_influencers) > 0)
        nnz_ixs = np.array(gradients_flipped.nonzero()).T

        sorting = np.argsort(gradients_flipped[tuple(nnz_ixs.T)]).A1
        sorted_ixs = nnz_ixs[sorting]
        grads = gradients_flipped[tuple(nnz_ixs[sorting].T)]

        scores = surrogate_loss - grads
        return sorted_ixs[::-1], scores.A1[::-1]

    def struct_score(self, a_hat_uv, XW):
        """
        Compute structure scores, cf. Eq. 15 in the paper

        Parameters
        ----------
        a_hat_uv: sp.sparse_matrix, shape [P, 2]
            Entries of matrix A_hat^2_u for each potential edge (see paper for explanation)

        XW: np.array, shape [num_nodes, num_classes], dtype float
            The class logits for each node.

        Returns
        -------
        np.array [P,]
            The struct score for every row in a_hat_uv
        """

        logits = a_hat_uv @ XW
        label_onehot = np.eye(self.num_classes)[self.target_label]
        best_wrong_class_logits = (logits - 1000 * label_onehot).max(1)
        logits_for_correct_class = logits[:, self.target_label]
        struct_scores = logits_for_correct_class - best_wrong_class_logits

        return struct_scores

    def compute_XW(self):
        """
        Shortcut to compute the dot product of X and W
        Returns
        -------
        x @ W: np.array, shape [num_nodes, num_classes]
        """

        return self.modified_nx @ self.W

    def get_attacker_nodes(self, n=5, add_additional_nodes=False):
        """
        Determine the influencer nodes to attack node i based on the weights W and the attributes X.

        Parameters
        ----------
        n: int, default: 5
            The desired number of attacker nodes.

        add_additional_nodes: bool, default: False
            if True and the degree of node i (d_u) is < n, we select n-d_u additional attackers, which should
            get connected to u afterwards (outside this function).

        Returns
        -------
        np.array, shape [n,]:
            The indices of the attacker nodes.
        optional: np.array, shape [n - degree(n)]
            if additional_nodes is True, we separately
            return the additional attacker node indices

        """

        assert n < self.num_nodes - 1, "number of influencers cannot be >= number of nodes in the graph!"

        #         neighbors = self.modified_adj[self.target].nonzero()[1]
        neighbors = self.modified_adj[self.target].indices
        #         assert self.target not in neighbors

        potential_edges = np.column_stack((np.tile(self.target, len(neighbors)), neighbors)).astype("int32")
        # The new A_hat_square_uv values that we would get if we removed the edge from u to each of the neighbors, respectively
        a_hat_uv = self.compute_new_a_hat_uv(potential_edges)

        XW = self.compute_XW()

        # compute the struct scores for all neighbors
        struct_scores = self.struct_score(a_hat_uv, XW)
        if len(neighbors) >= n:  # do we have enough neighbors for the number of desired influencers?
            influence_nodes = neighbors[np.argsort(struct_scores)[:n]]
            if add_additional_nodes:
                return influence_nodes, np.array([])
            return influence_nodes
        else:
            influence_nodes = neighbors
            if add_additional_nodes:  # Add additional influencers by connecting them to u first.
                # Compute the set of possible additional influencers, i.e. all nodes except the ones
                # that are already connected to u.
                poss_add_infl = np.setdiff1d(np.setdiff1d(np.arange(self.num_nodes), neighbors), self.target)
                n_possible_additional = len(poss_add_infl)
                n_additional_attackers = n - len(neighbors)
                possible_edges = np.column_stack((np.tile(self.target, n_possible_additional), poss_add_infl)).astype("int32")

                # Compute the struct_scores for all possible additional influencers, and choose the one
                # with the best struct score.
                a_hat_uv_additional = self.compute_new_a_hat_uv(possible_edges)
                additional_struct_scores = self.struct_score(a_hat_uv_additional, XW)
                additional_influencers = poss_add_infl[np.argsort(additional_struct_scores)[-n_additional_attackers::]]

                return influence_nodes, additional_influencers
            else:
                return influence_nodes

    def compute_new_a_hat_uv(self, potential_edges):
        """
        Compute the updated A_hat_square_uv entries that would result from inserting/deleting the input edges,
        for every edge.

        Parameters
        ----------
        potential_edges: np.array, shape [P,2], dtype int
            The edges to check.

        Returns
        -------
        sp.sparse_matrix: updated A_hat_square_u entries, a sparse PxN matrix, where P is len(possible_edges).
        """

        edges = np.transpose(self.modified_adj.nonzero())
        edges_set = {tuple(e) for e in edges}
        A_hat_sq = self.adj_norm @ self.adj_norm
        values_before = A_hat_sq[self.target].toarray()[0]
        node_ixs = np.unique(edges[:, 0], return_index=True)[1].astype("int32")
        twohop_ixs = np.transpose(A_hat_sq.nonzero())
        degrees = self.modified_adj.sum(0).A1 + 1

        # Ignore warnings:
        #     NumbaPendingDeprecationWarning:
        # Encountered the use of a type that is scheduled for deprecation: type 'reflected set' found for argument 'edges_set' of function 'compute_new_a_hat_uv'.

        # For more information visit http://numba.pydata.org/numba-doc/latest/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings(
                'ignore',
                '.*Encountered the use of a type that is scheduled for deprecation*'
            )
            ixs, vals = compute_new_a_hat_uv(edges, node_ixs, edges_set,
                                             twohop_ixs, values_before,
                                             degrees, potential_edges,
                                             self.target)
        ixs_arr = np.array(ixs)
        a_hat_uv = sp.coo_matrix((vals, (ixs_arr[:, 0], ixs_arr[:, 1])),
                                 shape=[len(potential_edges), self.num_nodes])

        return a_hat_uv

    def attack(self,
               target,
               num_budgets=None,
               direct_attack=True,
               structure_attack=True,
               feature_attack=False,
               n_influencers=5,
               ll_constraint=True,
               ll_cutoff=0.004,
               disable=False):

        super().attack(target, num_budgets, direct_attack, structure_attack,
                       feature_attack)

        if feature_attack and not self.graph.is_binary():
            raise RuntimeError(
                "Currently only attack binary node attributes are supported")

        if ll_constraint and self.allow_singleton:
            raise RuntimeError(
                '`ll_constraint` is failed when `allow_singleton=True`, please set `attacker.allow_singleton=False`.'
            )

        logits_start = self.compute_logits()
        best_wrong_class = self.strongest_wrong_class(logits_start)

        if structure_attack and ll_constraint:
            # Setup starting values of the likelihood ratio test.
            degree_sequence_start = self.degree
            current_degree_sequence = self.degree.astype('float64')
            d_min = 2
            S_d_start = np.sum(
                np.log(degree_sequence_start[degree_sequence_start >= d_min]))
            current_S_d = np.sum(
                np.log(
                    current_degree_sequence[current_degree_sequence >= d_min]))
            n_start = np.sum(degree_sequence_start >= d_min)
            current_n = np.sum(current_degree_sequence >= d_min)
            alpha_start = compute_alpha(n_start, S_d_start, d_min)
            log_likelihood_orig = compute_log_likelihood(
                n_start, alpha_start, S_d_start, d_min)

        if len(self.influence_nodes) == 0:
            if not direct_attack:
                # Choose influencer nodes
                infls, add_infls = self.get_attacker_nodes(
                    n_influencers, add_additional_nodes=True)
                self.influence_nodes = np.concatenate((infls, add_infls))
                # Potential edges are all edges from any attacker to any other node, except the respective
                # attacker itself or the node being attacked.
                self.potential_edges = np.row_stack([
                    np.column_stack(
                        (np.tile(infl, self.num_nodes - 2),
                         np.setdiff1d(np.arange(self.num_nodes),
                                      np.array([self.target, infl]))))
                    for infl in self.influence_nodes
                ])
            else:
                # direct attack
                influencers = [self.target]
                self.potential_edges = np.column_stack(
                    (np.tile(self.target, self.num_nodes - 1),
                     np.setdiff1d(np.arange(self.num_nodes), self.target)))
                self.influence_nodes = np.array(influencers)

        self.potential_edges = self.potential_edges.astype("int32")

        for it in tqdm(range(self.num_budgets),
                       desc='Peturbing Graph',
                       disable=disable):
            if structure_attack:
                # Do not consider edges that, if removed, result in singleton edges in the graph.
                if not self.allow_singleton:
                    filtered_edges = gf.filter_singletons(self.potential_edges, self.modified_adj).astype("int32")
                else:
                    filtered_edges = self.potential_edges

                if ll_constraint:
                    # Update the values for the power law likelihood ratio test.
                    deltas = 2 * (1 - self.modified_adj[tuple(
                        filtered_edges.T)].A.ravel()) - 1
                    d_edges_old = current_degree_sequence[filtered_edges]
                    d_edges_new = current_degree_sequence[
                        filtered_edges] + deltas[:, None]
                    new_S_d, new_n = update_Sx(current_S_d, current_n,
                                               d_edges_old, d_edges_new, d_min)
                    new_alphas = compute_alpha(new_n, new_S_d, d_min)
                    new_ll = compute_log_likelihood(new_n, new_alphas, new_S_d,
                                                    d_min)
                    alphas_combined = compute_alpha(new_n + n_start,
                                                    new_S_d + S_d_start, d_min)
                    new_ll_combined = compute_log_likelihood(
                        new_n + n_start, alphas_combined, new_S_d + S_d_start,
                        d_min)
                    new_ratios = -2 * new_ll_combined + 2 * (
                        new_ll + log_likelihood_orig)

                    # Do not consider edges that, if added/removed, would lead to a violation of the
                    # likelihood ration Chi_square cutoff value.
                    powerlaw_filter = filter_chisquare(new_ratios, ll_cutoff)
                    filtered_edges = filtered_edges[powerlaw_filter]

                # Compute new entries in A_hat_square_uv
                a_hat_uv_new = self.compute_new_a_hat_uv(filtered_edges)
                # Compute the struct scores for each potential edge
                struct_scores = self.struct_score(a_hat_uv_new,
                                                  self.compute_XW())
                best_edge_ix = struct_scores.argmin()
                best_edge_score = struct_scores.min()
                best_edge = filtered_edges[best_edge_ix]

            if feature_attack:
                # Compute the feature scores for each potential feature perturbation
                feature_ixs, feature_scores = self.feature_scores()
                best_feature_ix = feature_ixs[0]
                best_feature_score = feature_scores[0]

            if structure_attack and feature_attack:
                # decide whether to choose an edge or feature to change
                if best_edge_score < best_feature_score:
                    change_structure = True
                else:
                    change_structure = False

            elif structure_attack:
                change_structure = True
            elif feature_attack:
                change_structure = False

            if change_structure:
                # perform edge perturbation
                u, v = best_edge
                modified_adj = self.modified_adj.tolil(copy=False)
                modified_adj[(u, v)] = modified_adj[(
                    v, u)] = 1 - modified_adj[(u, v)]
                self.modified_adj = modified_adj.tocsr(copy=False)
                self.adj_norm = gf.normalize_adj(modified_adj)
                self.adj_flips.append((u, v))

                if ll_constraint:
                    # Update likelihood ratio test values
                    current_S_d = new_S_d[powerlaw_filter][best_edge_ix]
                    current_n = new_n[powerlaw_filter][best_edge_ix]
                    current_degree_sequence[best_edge] += deltas[
                        powerlaw_filter][best_edge_ix]
            else:

                modified_nx = self.modified_nx.tolil(copy=False)
                modified_nx[tuple(
                    best_feature_ix)] = 1 - modified_nx[tuple(best_feature_ix)]
                self.modified_nx = modified_nx.tocsr(copy=False)
                self.nattr_flips.append(tuple(best_feature_ix))
        return self


@njit
def connected_after(u, v, connected_before, delta):
    if u == v:
        if delta == -1:
            return False
        else:
            return True
    else:
        return connected_before


@njit
def compute_new_a_hat_uv(edge_ixs, node_nb_ixs, edges_set, twohop_ixs,
                         values_before, degs, potential_edges, u):
    """
    Compute the new values [A_hat_square]_u for every potential edge, where u is the target node. C.f. Theorem 5.1
    equation 17.

    Parameters
    ----------
    edge_ixs: np.array, shape [E,2], where E is the number of edges in the graph.
        The indices of the nodes connected by the edges in the input graph.
    node_nb_ixs: np.array, shape [num_nodes,], dtype int
        For each node, this gives the first index of edges associated to this node in the edge array (edge_ixs).
        This will be used to quickly look up the neighbors of a node, since numba does not allow nested lists.
    edges_set: set((e0, e1))
        The set of edges in the input graph, i.e. e0 and e1 are two nodes connected by an edge
    twohop_ixs: np.array, shape [T, 2], where T is the number of edges in A_tilde^2
        The indices of nodes that are in the twohop neighborhood of each other, including self-loops.
    values_before: np.array, shape [num_nodes,], the values in [A_hat]^2_uv to be updated.
    degs: np.array, shape [num_nodes,], dtype int
        The degree of the nodes in the input graph.
    potential_edges: np.array, shape [P, 2], where P is the number of potential edges.
        The potential edges to be evaluated. For each of these potential edges, this function will compute the values
        in [A_hat]^2_uv that would result after inserting/removing this edge.
    u: int
        The target node

    Returns
    -------
    return_ixs: List of tuples
        The ixs in the [P, num_nodes] matrix of updated values that have changed
    return_values:

    """
    num_nodes = degs.shape[0]

    twohop_u = twohop_ixs[twohop_ixs[:, 0] == u, 1]
    nbs_u = edge_ixs[edge_ixs[:, 0] == u, 1]
    nbs_u_set = set(nbs_u)

    return_ixs = []
    return_values = []

    for ix in range(len(potential_edges)):
        edge = potential_edges[ix]
        edge_set = set(edge)
        degs_new = degs.copy()
        delta = -2 * ((edge[0], edge[1]) in edges_set) + 1
        degs_new[edge] += delta

        nbs_edge0 = edge_ixs[edge_ixs[:, 0] == edge[0], 1]
        nbs_edge1 = edge_ixs[edge_ixs[:, 0] == edge[1], 1]

        affected_nodes = set(np.concatenate((twohop_u, nbs_edge0, nbs_edge1)))
        affected_nodes = affected_nodes.union(edge_set)
        a_um = edge[0] in nbs_u_set
        a_un = edge[1] in nbs_u_set

        a_un_after = connected_after(u, edge[0], a_un, delta)
        a_um_after = connected_after(u, edge[1], a_um, delta)

        for v in affected_nodes:
            a_uv_before = v in nbs_u_set
            a_uv_before_sl = a_uv_before or v == u

            if v in edge_set and u in edge_set and u != v:
                if delta == -1:
                    a_uv_after = False
                else:
                    a_uv_after = True
            else:
                a_uv_after = a_uv_before
            a_uv_after_sl = a_uv_after or v == u

            from_ix = node_nb_ixs[v]
            to_ix = node_nb_ixs[v + 1] if v < num_nodes - 1 else len(edge_ixs)
            node_nbs = edge_ixs[from_ix:to_ix, 1]
            node_nbs_set = set(node_nbs)
            a_vm_before = edge[0] in node_nbs_set

            a_vn_before = edge[1] in node_nbs_set
            a_vn_after = connected_after(v, edge[0], a_vn_before, delta)
            a_vm_after = connected_after(v, edge[1], a_vm_before, delta)

            mult_term = 1 / np.sqrt(degs_new[u] * degs_new[v])

            sum_term1 = np.sqrt(degs[u] * degs[v]) * values_before[v] - a_uv_before_sl / degs[u] - a_uv_before / \
                degs[v]
            sum_term2 = a_uv_after / degs_new[v] + a_uv_after_sl / degs_new[u]
            sum_term3 = -((a_um and a_vm_before) / degs[edge[0]]) + (
                a_um_after and a_vm_after) / degs_new[edge[0]]
            sum_term4 = -((a_un and a_vn_before) / degs[edge[1]]) + (
                a_un_after and a_vn_after) / degs_new[edge[1]]
            new_val = mult_term * (sum_term1 + sum_term2 + sum_term3 +
                                   sum_term4)

            return_ixs.append((ix, v))
            return_values.append(new_val)

    return return_ixs, return_values


def compute_alpha(n, S_d, d_min):
    """
    Approximate the alpha of a power law distribution.

    Parameters
    ----------
    n: int or np.array of int
        Number of entries that are larger than or equal to d_min

    S_d: float or np.array of float
         Sum of log degrees in the distribution that are larger than or equal to d_min

    d_min: int
        The minimum degree of nodes to consider

    Returns
    -------
    alpha: float
        The estimated alpha of the power law distribution
    """

    return n / (S_d - n * np.log(d_min - 0.5)) + 1


def update_Sx(S_old, n_old, d_old, d_new, d_min):
    """
    Update on the sum of log degrees S_d and n based on degree distribution resulting from inserting or deleting
    a single edge.

    Parameters
    ----------
    S_old: float
         Sum of log degrees in the distribution that are larger than or equal to d_min.

    n_old: int
        Number of entries in the old distribution that are larger than or equal to d_min.

    d_old: np.array, shape [num_nodes,] dtype int
        The old degree sequence.

    d_new: np.array, shape [num_nodes,] dtype int
        The new degree sequence

    d_min: int
        The minimum degree of nodes to consider

    Returns
    -------
    new_S_d: float, the updated sum of log degrees in the distribution that are larger than or equal to d_min.
    new_n: int, the updated number of entries in the old distribution that are larger than or equal to d_min.
    """

    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min

    d_old_in_range = np.multiply(d_old, old_in_range)
    d_new_in_range = np.multiply(d_new, new_in_range)

    new_S_d = S_old - np.log(np.maximum(d_old_in_range, 1)).sum(1) + np.log(
        np.maximum(d_new_in_range, 1)).sum(1)
    new_n = n_old - np.sum(old_in_range, 1) + np.sum(new_in_range, 1)

    return new_S_d, new_n


def compute_log_likelihood(n, alpha, S_d, d_min):
    """
    Compute log likelihood of the powerlaw fit.

    Parameters
    ----------
    n: int
        Number of entries in the old distribution that are larger than or equal to d_min.

    alpha: float
        The estimated alpha of the power law distribution

    S_d: float
         Sum of log degrees in the distribution that are larger than or equal to d_min.

    d_min: int
        The minimum degree of nodes to consider

    Returns
    -------
    float: the estimated log likelihood
    """

    return n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * S_d


def filter_chisquare(ll_ratios, cutoff):
    return ll_ratios < cutoff
