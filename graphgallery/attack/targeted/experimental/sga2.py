import numpy as np
import tensorflow as tf
from numba import njit

from tensorflow.keras.losses import sparse_categorical_crossentropy
from graphadv.attack.targeted.targeted_attacker import TargetedAttacker
from graphadv.utils.surrogate_utils import train_a_surrogate

from graphgallery.nn.layers import SGConv
from graphgallery.nn.models import SGC
from graphgallery import tqdm, astensor, ego_graph


class SGA(TargetedAttacker):
    def __init__(self,
                 adj,
                 x,
                 labels,
                 idx_train=None,
                 hops=2,
                 seed=None,
                 name=None,
                 device='CPU:0',
                 surrogate=None,
                 surrogate_args={},
                 **kwargs):

        super().__init__(adj,
                         x=x,
                         labels=labels,
                         seed=seed,
                         name=name,
                         device=device,
                         **kwargs)

        if surrogate is None:
            surrogate = train_a_surrogate(self, 'SGC', idx_train,
                                          **surrogate_args)
        elif not isinstance(surrogate, SGC):
            raise RuntimeError(
                "surrogate model should be the instance of `graphgallery.nn.SGC`."
            )

        self.hops = hops
        self.similar_nodes = [
            np.where(labels == class_)[0] for class_ in range(self.num_classes)
        ]

        with tf.device(self.device):
            W, b = surrogate.weights
            X = astensor(x)
            self.b = b
            self.XW = X @ W
            self.surrogate = surrogate
            self.SGC = SGConv(hops)
            self.loss_fn = sparse_categorical_crossentropy

    def reset(self):
        super().reset()
        # for the added self-loop
        self.selfloop_degree = (self.degree + 1.).astype(self.floatx)
        self.adj_flips = {}
        self.pos_dict = None
        self.wrong_label = None

    def attack(self,
               target,
               num_budgets=None,
               reduce_nodes=3,
               direct_attack=True,
               structure_attack=True,
               feature_attack=False,
               compute_A_grad=True,
               disable=False):

        super().attack(target, num_budgets, direct_attack, structure_attack,
                       feature_attack)

        logit = self.surrogate.predict(target).ravel()
        top2 = logit.argsort()[-2:]
        self.wrong_label = top2[-1] if top2[-1] != self.target_label else top2[
            0]
        assert self.wrong_label != self.target_label
        self.subgraph_preprocessing(reduce_nodes)

        offset = self.edge_lower_bound
        weights = self.weights
        with tf.device(self.device):
            for _ in tqdm(range(self.num_budgets),
                          desc='Peturbing Graph',
                          disable=disable):
                gradients = self.compute_gradient(
                    compute_A_grad=compute_A_grad)
                gradients *= (-2. * weights) + 1.
                gradients = gradients[offset:]
                sorted_index = tf.argsort(gradients, direction='DESCENDING')

                for index in sorted_index:
                    index_with_offset = index + offset
                    u, v = self.indices[index_with_offset]
                    if index_with_offset < self.non_edge_lower_bound and not self.allow_singleton and (
                            self.selfloop_degree[u] <= 2
                            or self.selfloop_degree[v] <= 2):
                        continue

                    if not self.is_modified(u, v):
                        self.adj_flips[(u, v)] = index_with_offset
                        self.update_subgraph(u, v, index_with_offset)
                        break

    def subgraph_preprocessing(self, reduce_nodes):

        target = self.target
        wrong_label = self.wrong_label
        # neighbors = self.graph.adj_matrix[target].nonzero()[1]
        neighbors = self.graph.adj_matrix[target].indices
        wrong_label_nodes = self.similar_nodes[wrong_label]
        sub_edges, sub_nodes = self.ego_subgraph()
        sub_edges = np.vstack([sub_edges, sub_edges[:, [1, 0]]])

        if self.direct_attack or reduce_nodes is not None:
            influence_nodes = [target]
            wrong_label_nodes = np.setdiff1d(wrong_label_nodes, neighbors)
        else:
            influence_nodes = neighbors

        self.construct_sub_adj(influence_nodes, wrong_label_nodes, sub_nodes,
                               sub_edges)

        if reduce_nodes is not None:
            if self.direct_attack:
                influence_nodes = [target]
                wrong_label_nodes = self.top_k_wrong_labels_nodes(
                    k=self.num_budgets)

            else:
                influence_nodes = neighbors
                wrong_label_nodes = self.top_k_wrong_labels_nodes(
                    k=reduce_nodes)

            self.construct_sub_adj(influence_nodes, wrong_label_nodes,
                                   sub_nodes, sub_edges)

    @tf.function
    def SGC_conv(self, XW, adj):
        return self.SGC([XW, adj])

    def compute_gradient(self, eps=2.24, compute_A_grad=False):

        weights = self.weights
        if not compute_A_grad:
            weights = normalize_GCN(self.indices, weights,
                                    self.selfloop_degree)

        with tf.GradientTape() as tape:
            tape.watch(weights)

            if not compute_A_grad:
                adj = tf.sparse.SparseTensor(self.indices, weights,
                                             self.graph.adj_matrix.shape)
            else:
                weights_norm = normalize_GCN(self.indices, weights,
                                             self.selfloop_degree)
                adj = tf.sparse.SparseTensor(self.indices, weights_norm,
                                             self.graph.adj_matrix.shape)

            output = self.SGC_conv(self.XW, adj)
            logit = tf.nn.softmax(((output[self.target] + self.b) / eps))
            loss = self.loss_fn(self.target_label, logit) - self.loss_fn(
                self.wrong_label, logit)

        gradients = tape.gradient(loss, weights)

        return gradients

    def ego_subgraph(self):
        return ego_graph(self.graph.adj_matrix, self.target, self.hops)

    def construct_sub_adj(self, influence_nodes, wrong_label_nodes, sub_nodes,
                          sub_edges):
        length = len(wrong_label_nodes)
        potential_edges = np.vstack([
            np.stack([np.tile(infl, length), wrong_label_nodes], axis=1)
            for infl in influence_nodes
        ])

        if len(influence_nodes) > 1:
            # TODO: considering self-loops
            mask = self.graph.adj_matrix[tuple(potential_edges.T)].A1 == 0
            potential_edges = potential_edges[mask]

        nodes = np.union1d(sub_nodes, wrong_label_nodes)
        edge_weights = np.ones(sub_edges.shape[0], dtype=self.floatx)
        non_edge_weights = np.zeros(potential_edges.shape[0],
                                    dtype=self.floatx)
        self_loop_weights = np.ones(nodes.shape[0], dtype=self.floatx)
        self_loop = np.stack([nodes, nodes], axis=1)

        self.indices = np.vstack([
            self_loop, potential_edges[:, [1, 0]], sub_edges, potential_edges
        ])
        weights = np.hstack([
            self_loop_weights, non_edge_weights, edge_weights, non_edge_weights
        ])
        self.weights = tf.Variable(weights, dtype=self.floatx)
        self.edge_lower_bound = self_loop_weights.size + non_edge_weights.size
        self.non_edge_lower_bound = self.edge_lower_bound + edge_weights.size

        self.n_sub_edges = edge_weights.size // 2
        self.n_non_edges = non_edge_weights.size

    def top_k_wrong_labels_nodes(self, k):
        offset = self.non_edge_lower_bound
        weights = self.weights
        with tf.device(self.device):
            gradients = self.compute_gradient()[offset:]
            _, index = tf.math.top_k(gradients, k=k)

        wrong_label_nodes = self.indices[:, 1][index.numpy() + offset]

        return wrong_label_nodes

    def update_subgraph(self, u, v, index):
        weight = 1.0 - self.weights[index]
        degree_delta = 2. * weight - 1.
        if weight > 0:
            inv_index = index - self.n_non_edges - self.n_sub_edges * 2
        else:
            if index >= self.edge_lower_bound + self.n_sub_edges:
                inv_index = index - self.n_sub_edges
            else:
                inv_index = index + self.n_sub_edges

        self.weights[index].assign(weight)
        self.weights[inv_index].assign(weight)
        self.selfloop_degree[u] += degree_delta
        self.selfloop_degree[v] += degree_delta


def normalize_GCN(indices, weights, degree):
    row, col = indices.T
    inv_degree = tf.pow(degree, -0.5)
    normed_weights = weights * tf.gather(inv_degree, row) * tf.gather(
        inv_degree, col)
    return normed_weights
