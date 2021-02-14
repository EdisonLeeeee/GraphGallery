import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy

import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.utils import tqdm
from graphgallery.nn.layers.tensorflow import SGConv
from graphgallery.attack.targeted import TensorFlow
from ..targeted_attacker import TargetedAttacker


@TensorFlow.register()
class SGA(TargetedAttacker):
    def process(self, surrogate, reset=True):
        assert isinstance(surrogate, gg.gallery.nodeclas.SGC), surrogate

        hops = surrogate.cfg.process.K  # NOTE: Be compatible with graphgallery
        # nodes with the same class labels
        self.similar_nodes = [
            np.where(self.graph.node_label == c)[0]
            for c in range(self.num_classes)
        ]

        with tf.device(self.device):
            W, b = surrogate.model.weights
            X = gf.astensor(self.graph.node_attr)
            self.b = b
            self.XW = X @ W
            self.SGC = SGConv(hops)
            self.hops = hops
            self.loss_fn = sparse_categorical_crossentropy
            self.surrogate = surrogate
        if reset:
            self.reset()
        return self

    def reset(self):
        super().reset()
        # for the added self-loop
        self.selfloop_degree = (self.degree + 1.).astype(self.floatx)
        self.adj_flips = {}
        self.wrong_label = None
        return self

    def attack(self,
               target,
               num_budgets=None,
               logit=None,
               reduced_nodes=3,
               direct_attack=True,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        super().attack(target, num_budgets, direct_attack, structure_attack,
                       feature_attack)

        if logit is None:
            logit = self.surrogate.predict(target).ravel()

        top2 = logit.argsort()[-2:]
        self.wrong_label = np.setdiff1d(top2, self.target_label)[0]
        assert self.wrong_label != self.target_label

        self.subgraph_preprocessing(reduced_nodes)
        offset = self.edge_weights.shape[0]

        with tf.device(self.device):
            for it in tqdm(range(self.num_budgets),
                           desc='Peturbing Graph',
                           disable=disable):
                edge_grad, non_edge_grad = self.compute_gradient(norm=False)
                edge_grad *= (-2 * self.edge_weights + 1)
                non_edge_grad *= (-2 * self.non_edge_weights + 1)
                gradients = tf.concat([edge_grad, non_edge_grad], axis=0)
                sorted_indices = tf.argsort(gradients, direction="DESCENDING")

                for index in sorted_indices:
                    if index < offset:
                        u, v = self.edge_index[:, index]
                        add = False
                        if not self.allow_singleton and (
                                self.selfloop_degree[u] <= 2
                                or self.selfloop_degree[v] <= 2):
                            continue
                    else:
                        index -= offset
                        u, v = self.non_edge_index[:, index]
                        add = True

                    if not self.is_modified(u, v):
                        self.adj_flips[(u, v)] = it
                        self.update_subgraph(u, v, index, add=add)
                        break
        return self

    def subgraph_preprocessing(self, reduced_nodes=None):
        target = self.target
        wrong_label = self.wrong_label
        neighbors = self.graph.adj_matrix[target].indices
        wrong_label_nodes = self.similar_nodes[wrong_label]
        sub_edges, sub_nodes = self.ego_subgraph()
        sub_edges = sub_edges.T  # shape [2, M]

        if self.direct_attack or reduced_nodes is not None:
            influence_nodes = [target]
            wrong_label_nodes = np.setdiff1d(wrong_label_nodes, neighbors)
        else:
            influence_nodes = neighbors

        self.construct_sub_adj(influence_nodes, wrong_label_nodes, sub_nodes,
                               sub_edges)

        if reduced_nodes is not None:
            if self.direct_attack:
                influence_nodes = [target]
                wrong_label_nodes = self.top_k_wrong_labels_nodes(
                    k=self.num_budgets)

            else:
                influence_nodes = neighbors
                wrong_label_nodes = self.top_k_wrong_labels_nodes(
                    k=reduced_nodes)

            self.construct_sub_adj(influence_nodes, wrong_label_nodes,
                                   sub_nodes, sub_edges)

    @tf.function
    def SGC_conv(self, XW, adj):
        return self.SGC([XW, adj])

    def compute_gradient(self, eps=5.0, norm=False):

        edge_weights = self.edge_weights
        non_edge_weights = self.non_edge_weights
        self_loop_weights = self.self_loop_weights

        if norm:
            edge_weights = normalize_GCN(self.edge_index, edge_weights,
                                         self.selfloop_degree)
            non_edge_weights = normalize_GCN(self.non_edge_index,
                                             non_edge_weights,
                                             self.selfloop_degree)
            self_loop_weights = normalize_GCN(self.self_loop,
                                              self_loop_weights,
                                              self.selfloop_degree)

        with tf.GradientTape() as tape:
            tape.watch([edge_weights, non_edge_weights])

            weights = tf.concat([
                edge_weights, edge_weights, non_edge_weights, non_edge_weights,
                self_loop_weights
            ],
                axis=0)

            if norm:
                adj = tf.sparse.SparseTensor(self.indices.T, weights,
                                             self.graph.adj_matrix.shape)
            else:
                weights_norm = normalize_GCN(self.indices, weights,
                                             self.selfloop_degree)
                adj = tf.sparse.SparseTensor(self.indices.T, weights_norm,
                                             self.graph.adj_matrix.shape)

            output = self.SGC_conv(self.XW, adj)
            logit = output[self.target] + self.b
            # model calibration
            logit = tf.nn.softmax(logit / eps)
            loss = self.loss_fn(self.target_label, logit) - self.loss_fn(
                self.wrong_label, logit)

        gradients = tape.gradient(loss, [edge_weights, non_edge_weights])
        return gradients

    def ego_subgraph(self):
        return gf.ego_graph(self.graph.adj_matrix, self.target, self.hops)

    def construct_sub_adj(self, influence_nodes, wrong_label_nodes, sub_nodes,
                          sub_edges):
        length = len(wrong_label_nodes)
        potential_edges = np.hstack([
            np.row_stack([np.tile(infl, length), wrong_label_nodes])
            for infl in influence_nodes
        ])

        if len(influence_nodes) > 1:
            # TODO: considering self-loops
            mask = self.graph.adj_matrix[potential_edges[0],
                                         potential_edges[1]].A1 == 0
            potential_edges = potential_edges[:, mask]

        nodes = np.union1d(sub_nodes, wrong_label_nodes)
        edge_weights = np.ones(sub_edges.shape[1], dtype=self.floatx)
        non_edge_weights = np.zeros(potential_edges.shape[1],
                                    dtype=self.floatx)
        self_loop_weights = np.ones(nodes.shape[0], dtype=self.floatx)
        self_loop = np.row_stack([nodes, nodes])

        self.indices = np.hstack([
            sub_edges, sub_edges[[1, 0]], potential_edges,
            potential_edges[[1, 0]], self_loop
        ])
        self.edge_weights = tf.Variable(edge_weights, dtype=self.floatx)
        self.non_edge_weights = tf.Variable(non_edge_weights,
                                            dtype=self.floatx)
        self.self_loop_weights = gf.astensor(self_loop_weights,
                                             dtype=self.floatx)
        self.edge_index = sub_edges
        self.non_edge_index = potential_edges
        self.self_loop = self_loop

    def top_k_wrong_labels_nodes(self, k):
        with tf.device(self.device):
            _, non_edge_grad = self.compute_gradient(norm=True)
            _, index = tf.math.top_k(non_edge_grad, k=k, sorted=False)

        wrong_label_nodes = self.non_edge_index[1][index.numpy()]
        return wrong_label_nodes

    def update_subgraph(self, u, v, index, add=True):
        if add:
            self.non_edge_weights[index].assign(1.0)
            degree_delta = 1.0
        else:
            self.edge_weights[index].assign(0.0)
            degree_delta = -1.0

        self.selfloop_degree[u] += degree_delta
        self.selfloop_degree[v] += degree_delta


def normalize_GCN(indices, weights, degree):
    row, col = indices
    inv_degree = tf.pow(degree, -0.5)
    normed_weights = weights * tf.gather(inv_degree, row) * tf.gather(
        inv_degree, col)
    return normed_weights
