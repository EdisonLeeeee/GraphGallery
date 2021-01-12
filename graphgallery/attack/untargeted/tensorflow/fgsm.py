import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy

import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.utils import tqdm
from graphgallery.attack.untargeted import TensorFlow
from ..untargeted_attacker import UntargetedAttacker


@TensorFlow.register()
class FGSM(UntargetedAttacker):
    _allow_feature_attack = True

    def process(self, surrogate, victim_nodes, victim_labels=None, reset=True):
        if isinstance(surrogate, gg.gallery.Trainer):
            surrogate = surrogate.model

        if victim_labels is None:
            victim_labels = self.graph.node_label[victim_nodes]

        with tf.device(self.device):
            self.surrogate = surrogate
            self.loss_fn = sparse_categorical_crossentropy
            self.victim_nodes = gf.astensor(victim_nodes)
            self.victim_labels = gf.astensor(victim_labels)
        if reset:
            self.reset()
        return self

    def reset(self):
        super().reset()
        self.adj_flips = []
        self.nattr_flips = []

        with tf.device(self.device):
            self.modified_adj = tf.Variable(self.graph.adj_matrix.A,
                                            dtype=self.floatx)
            self.modified_nx = tf.Variable(self.graph.node_attr,
                                           dtype=self.floatx)
        return self

    def attack(self,
               num_budgets=0.05,
               symmetric=True,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        super().attack(num_budgets, structure_attack, feature_attack)

        if feature_attack and not self.graph.is_binary():
            raise RuntimeError(
                "Currently only attack binary node attributes are supported")

        modified_adj, modified_nx = self.modified_adj, self.modified_nx

        for it in tqdm(range(self.num_budgets),
                       desc='Peturbing Graph',
                       disable=disable):

            with tf.device(self.device):
                adj_grad, x_grad = self.compute_gradients(
                    modified_adj, modified_nx, self.victim_nodes, self.victim_labels)

                adj_grad_score = tf.constant(0.0)
                x_grad_score = tf.constant(0.0)

                if structure_attack:

                    if symmetric:
                        adj_grad += tf.transpose(adj_grad)

                    adj_grad_score = self.structure_score(
                        modified_adj, adj_grad)

                if feature_attack:
                    x_grad_score = self.feature_score(modified_nx, x_grad)

                if tf.reduce_max(adj_grad_score) >= tf.reduce_max(
                        x_grad_score):
                    adj_grad_argmax = tf.argmax(adj_grad_score)
                    row, col = divmod(adj_grad_argmax.numpy(), self.num_nodes)
                    modified_adj[row, col].assign(1. - modified_adj[row, col])
                    modified_adj[col, row].assign(1. - modified_adj[col, row])
                    self.adj_flips.append((row, col))
                else:
                    x_grad_argmax = tf.argmax(x_grad_score)
                    row, col = divmod(x_grad_argmax.numpy(), self.num_attrs)
                    modified_nx[row, col].assign(1. - modified_nx[row, col])
                    self.nattr_flips.append((row, col))
        return self

    @tf.function
    def structure_score(self, modified_adj, adj_grad):
        adj_grad = adj_grad * (-2 * modified_adj + 1)
        # Make sure that the minimum entry is 0.
        adj_grad = adj_grad - tf.reduce_min(adj_grad)
        # Filter self-loops
        adj_grad = adj_grad - tf.linalg.band_part(adj_grad, 0, 0)

        if not self.allow_singleton:
            # Set entries to 0 that could lead to singleton nodes.
            singleton_mask = self.filter_potential_singletons(modified_adj)
            adj_grad = adj_grad * singleton_mask

        return tf.reshape(adj_grad, [-1])

    @tf.function
    def feature_score(self, modified_nx, x_grad):

        x_grad = x_grad * (-2. * modified_nx + 1.)
        min_grad = tf.reduce_min(x_grad)
        x_grad = x_grad - min_grad

        return tf.reshape(x_grad, [-1])

    @tf.function
    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.
        Returns
        -------
        tf.Tensor shape [N, N], float with ones everywhere except the entries of potential singleton nodes,
        where the returned tensor has value 0.
        """
        N = self.num_nodes
        degrees = tf.reduce_sum(modified_adj, axis=1)
        degree_one = tf.equal(degrees, 1)
        resh = tf.reshape(tf.tile(degree_one, [N]), [N, N])
        l_and = tf.logical_and(resh, tf.equal(modified_adj, 1))
        logical_and_symmetric = tf.logical_or(l_and, tf.transpose(l_and))
        flat_mask = 1. - tf.cast(logical_and_symmetric, self.floatx)
        return flat_mask

    @tf.function
    def compute_gradients(self, modified_adj, modified_nx, victim_nodes, victim_labels):
        # TODO persistent=False
        persistent = self.structure_attack and self.feature_attack
        with tf.GradientTape(persistent=persistent) as tape:
            adj_norm = gf.normalize_adj_tensor(modified_adj)
            logit = self.surrogate([modified_nx, adj_norm])
            logit = tf.gather(logit, victim_nodes)
            loss = self.loss_fn(victim_labels, logit, from_logits=True)

        adj_grad, x_grad = None, None
        if self.structure_attack:
            adj_grad = tape.gradient(loss, modified_adj)

        if self.feature_attack:
            x_grad = tape.gradient(loss, modified_nx)

        return adj_grad, x_grad
