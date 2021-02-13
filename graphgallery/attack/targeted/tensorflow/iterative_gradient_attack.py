import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy

import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.utils import tqdm
from graphgallery.attack.targeted import TensorFlow
from ..targeted_attacker import TargetedAttacker


@TensorFlow.register()
class IGA(TargetedAttacker):
    '''Iterative Gradient Attack'''

    # IG can conduct feature attack
    _allow_feature_attack = True

    def process(self, surrogate, reset=True):
        if isinstance(surrogate, gg.gallery.nodeclas.Trainer):
            surrogate = surrogate.model

        with tf.device(self.device):
            self.surrogate = surrogate
            self.loss_fn = sparse_categorical_crossentropy

        if reset:
            self.reset()
        return self

    def reset(self):
        super().reset()
        self.target_index = None
        self.adj_flips = []
        self.nattr_flips = []

        with tf.device(self.device):
            self.modified_adj = tf.Variable(self.graph.adj_matrix.A,
                                            dtype=self.floatx)
            self.modified_nx = tf.Variable(self.graph.node_attr,
                                           dtype=self.floatx)
        return self

    def attack(self,
               target,
               num_budgets=None,
               symmetric=True,
               direct_attack=True,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        super().attack(target, num_budgets, direct_attack, structure_attack,
                       feature_attack)

        if feature_attack and not self.graph.is_binary():
            raise RuntimeError(
                "Currently only attack binary node attributes are supported")

        with tf.device(self.device):
            target_index = gf.astensor([self.target])
            target_labels = gf.astensor(self.target_label)

            modified_adj, modified_nx = self.modified_adj, self.modified_nx

            if not direct_attack:
                adj_mask, x_mask = self.construct_mask()
            else:
                adj_mask, x_mask = None, None

            for _ in tqdm(range(self.num_budgets),
                          desc='Peturbing Graph',
                          disable=disable):

                adj_grad, x_grad = self.compute_gradients(
                    modified_adj, modified_nx, target_index, target_labels)

                adj_grad_score = tf.constant(0.0)
                x_grad_score = tf.constant(0.0)

                if structure_attack:

                    if symmetric:
                        adj_grad = (adj_grad + tf.transpose(adj_grad)) / 2.

                    adj_grad_score = self.structure_score(
                        modified_adj, adj_grad, adj_mask)

                if feature_attack:
                    x_grad_score = self.feature_score(modified_nx, x_grad,
                                                      x_mask)

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

    def construct_mask(self):
        adj_mask = np.ones(self.graph.adj_matrix.shape, dtype=self.floatx)
        x_mask = np.ones(self.graph.node_attr.shape, dtype=self.floatx)
        adj_mask[:, self.target] = 0.
        adj_mask[self.target, :] = 0.
        x_mask[self.target, :] = 0

        adj_mask = gf.astensor(adj_mask)
        x_mask = gf.astensor(x_mask)

        return adj_mask, x_mask

    @tf.function
    def structure_score(self, modified_adj, adj_grad, adj_mask):
        adj_grad = adj_grad * (-2 * modified_adj + 1)
        # Make sure that the minimum entry is 0.
        adj_grad = adj_grad - tf.reduce_min(adj_grad)
        # Filter self-loops
        adj_grad = adj_grad - tf.linalg.band_part(adj_grad, 0, 0)

        if not self.allow_singleton:
            # Set entries to 0 that could lead to singleton nodes.
            singleton_mask = self.filter_potential_singletons(modified_adj)
            adj_grad = adj_grad * singleton_mask

        if not self.direct_attack:
            adj_grad = adj_grad * adj_mask

        return tf.reshape(adj_grad, [-1])

    @tf.function
    def feature_score(self, modified_nx, x_grad, x_mask):

        x_grad = x_grad * (-2. * modified_nx + 1.)
        x_grad = x_grad - tf.reduce_min(x_grad)

        if not self.direct_attack:
            x_grad = x_grad * x_mask

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
    def compute_gradients(self, modified_adj, modified_nx, target_index,
                          target_label):

        with tf.GradientTape(persistent=True) as tape:
            adj_norm = gf.normalize_adj_tensor(modified_adj)
            logit = self.surrogate([modified_nx, adj_norm])
            logit = tf.gather(logit, target_index)
            loss = self.loss_fn(target_label, logit, from_logits=True)

        adj_grad, x_grad = None, None

        if self.structure_attack:
            adj_grad = tape.gradient(loss, modified_adj)

        if self.feature_attack:
            x_grad = tape.gradient(loss, modified_nx)

        return adj_grad, x_grad
