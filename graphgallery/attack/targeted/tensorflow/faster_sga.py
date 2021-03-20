import numpy as np
import tensorflow as tf

from distutils.version import LooseVersion
if LooseVersion(tf.__version__) > LooseVersion("1.14"):
    import tensorflow.compat.v1 as tf

from graphgallery import functional as gf
from graphgallery.utils import tqdm
from graphgallery.attack.targeted import TensorFlow
from ..targeted_attacker import TargetedAttacker


@TensorFlow.register()
class FasterSGA(TargetedAttacker):
    """TensorFlow 1.x Implementation of SGA"""

    def process(self, W, b, K=2, normalize_grad=True, reset=True):
        # nodes with the same class labels
        self.similar_nodes = [
            np.where(self.graph.node_label == c)[0]
            for c in range(self.num_classes)
        ]
        W, b = gf.tensoras(W, b)
        self.K = K
        self.normalize_grad = normalize_grad
        self.surrogate = Surrogate(self.graph.node_attr @ W, b, K=K)
        self.shape = (self.num_nodes, self.num_nodes)
        self.adj = self.graph.adj_matrix
        edges, weights = gf.sparse_adj_to_edge(gf.normalize_adj(self.graph.adj_matrix))
        self.adj_sparse = (edges.T, weights, self.shape)
        self.y_onehot = np.eye(int(self.num_classes))[self.graph.node_label]

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

    def subgraph_preprocessing(self, target, node_reduction=True):
        logit = self.surrogate.run(self.surrogate.logit, feed_dict={self.surrogate.adj: self.adj_sparse,
                                                                    self.surrogate.target: target})
        self.label_onehot = self.y_onehot[target]
        self.wrong_label = np.argmax(logit - 1e6 * self.label_onehot)
        self.wrong_label_onehot = np.eye(self.num_classes)[self.wrong_label]
        self.edges, self.nodes = self.ego_subgraph()
        assert self.wrong_label != self.graph.node_label[target]

        neighbors = np.setdiff1d(self.adj[target].indices, target)
        self.neighbors = neighbors
        if self.direct_attack:
            influence_nodes = [target]
            nodes_with_wrong_label = np.setdiff1d(self.similar_nodes[self.wrong_label], neighbors + [target])
        else:
            if node_reduction:
                influence_nodes = [target]
            else:
                influence_nodes = neighbors

            nodes_with_wrong_label = np.setdiff1d(self.similar_nodes[self.wrong_label], [target])

        self.construct_sub_adj(nodes_with_wrong_label, influence_nodes)

        if node_reduction:
            if self.direct_attack:
                self.node_reduction([target], max_nodes=int(self.selfloop_degree[target]))
            else:
                self.node_reduction(neighbors, max_nodes=5)

    def ego_subgraph(self):
        edges, nodes = gf.ego_graph(self.adj, self.target, self.K)
        return edges, nodes

    def construct_sub_adj(self, nodes_with_wrong_label, influence_nodes):
        length = len(nodes_with_wrong_label)
        non_edge = np.vstack([np.stack([np.tile(infl, length), nodes_with_wrong_label], axis=1) for infl in influence_nodes])

        if len(influence_nodes) > 1:
            mask = self.adj[non_edge[0], non_edge[1]].A1 == 0
            non_edge = non_edge[mask]

        nodes_all = np.union1d(self.nodes, nodes_with_wrong_label)
        edge_weight = np.ones(len(self.edges), dtype=np.float32)
        non_edge_weight = np.zeros(len(non_edge), dtype=np.float32)

        self_loop = np.stack([nodes_all, nodes_all], axis=1)
        self_loop_weight = np.ones(nodes_all.size)

        self.indices = np.vstack([self.edges, non_edge, self.edges[:, [1, 0]], non_edge[:, [1, 0]], self_loop])
        self.upper_bound = edge_weight.size + non_edge_weight.size
        self.lower_bound = edge_weight.size
        self.non_edge = non_edge

        self.edge_weight = edge_weight
        self.non_edge_weight = non_edge_weight
        self.self_loop_weight = self_loop_weight

    def node_reduction(self, influence_nodes, max_nodes):
        sym_weights = np.hstack([self.edge_weight, self.non_edge_weight, self.edge_weight, self.non_edge_weight, self.self_loop_weight])
        norm_weight = normalize(sym_weights, self.indices, self.selfloop_degree)
        adj_norm = (self.indices, norm_weight, self.shape)
        feed_dict = self.surrogate.construct_feed_dict(adj_norm, self.label_onehot, self.wrong_label_onehot, self.target)
        gradients = self.surrogate.run(self.surrogate.gradients, feed_dict=feed_dict)[self.lower_bound:self.upper_bound]
        index = gf.least_indices(gradients, max_nodes)[0]
        self.construct_sub_adj(self.non_edge[index][:, 1], influence_nodes)

    def update_subgraph(self, u, v, idx):
        if idx < self.lower_bound:
            # remove edge
            degree_delta = -1
            self.edge_weight[idx] = 0.
        else:
            # add edge
            degree_delta = 1
            self.non_edge_weight[idx - self.lower_bound] = 1.0

        self.selfloop_degree[u] += degree_delta
        self.selfloop_degree[v] += degree_delta

    def attack(self,
               target,
               num_budgets=None,
               direct_attack=True,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        super().attack(target, num_budgets, direct_attack, structure_attack,
                       feature_attack)
        self.subgraph_preprocessing(target)
        for epoch in tqdm(range(self.num_budgets),
                          desc='Peturbing Graph',
                          disable=disable):
            weights = np.hstack([self.edge_weight, self.non_edge_weight])
            sym_weights = np.hstack([weights, weights, self.self_loop_weight])
            norm_weight = normalize(sym_weights, self.indices, self.selfloop_degree)
            adj_norm = (self.indices, norm_weight, self.shape)
            feed_dict = self.surrogate.construct_feed_dict(adj_norm, self.label_onehot, self.wrong_label_onehot, target)
            gradients = self.surrogate.run(self.surrogate.gradients, feed_dict=feed_dict)
            # a trick
            if self.normalize_grad:
                gradients = normalize(gradients, self.indices, self.selfloop_degree)

            gradients = gradients[:self.upper_bound] * (-2 * weights + 1)
            i = np.argmin(gradients)
            u, v = self.indices[i]
            assert not self.is_modified(u, v)
            self.adj_flips[(u, v)] = epoch
            self.update_subgraph(u, v, i)
        return self


def normalize(data, indices, degree):
    d = np.sqrt(degree)
    row, col = indices.T
    return data / (d[row] * d[col])


class Surrogate:
    def __init__(self, XW, b, K=2, eps=5.0):
        graph = tf.Graph()
        with graph.as_default():
            self.adj = tf.sparse_placeholder(dtype=tf.float32)
            self.label = tf.placeholder(dtype=tf.float32)
            self.wrong_label = tf.placeholder(dtype=tf.float32)
            self.target = tf.placeholder(dtype=tf.int32)

            XW = tf.constant(XW, dtype=tf.float32)
            b = tf.constant(b, dtype=tf.float32)

            out = XW
            for _ in range(K):
                out = tf.sparse.sparse_dense_matmul(self.adj, out)

            self.logit = out[self.target] + b

            # Calibration
            self.logit_calibrated = self.logit / eps

            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logit_calibrated, labels=self.wrong_label) - tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logit_calibrated, labels=self.label)
            self.gradients = tf.gradients(self.loss, self.adj.values)[0]
            self.sess = tf.Session(graph=graph)
            self.run(tf.global_variables_initializer())

    def construct_feed_dict(self, adj, label, wrong_label, target):
        feed_dict = {
            self.adj: adj,
            self.wrong_label: wrong_label,
            self.label: label,
            self.target: target,
        }
        return feed_dict

    def run(self, variables, feed_dict=None):
        return self.sess.run(variables, feed_dict=feed_dict)

    def close(self):
        self.sess.close()
