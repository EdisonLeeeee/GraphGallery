import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform, zeros
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax, relu
from tensorflow.keras.losses import SparseCategoricalCrossentropy

import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.utils import tqdm
# from graphadv.utils.graph_utils import likelihood_ratio_filter
from graphgallery.attack.untargeted import TensorFlow
from ..untargeted_attacker import UntargetedAttacker


class BaseMeta(UntargetedAttacker):
    """Base model for Mettack."""
    # mettack can also conduct feature attack
    _allow_feature_attack = True

    def process(self,
                train_nodes,
                unlabeled_nodes,
                self_training_labels,
                hids,
                use_relu,
                reset=True):

        self.ll_ratio = None

        with tf.device(self.device):
            self.train_nodes = gf.astensor(train_nodes, dtype=self.intx)
            self.unlabeled_nodes = gf.astensor(unlabeled_nodes, dtype=self.intx)
            self.labels_train = gf.astensor(self.graph.node_label[train_nodes], dtype=self.intx)
            self.self_training_labels = gf.astensor(self_training_labels, dtype=self.intx)
            self.adj_tensor = gf.astensor(self.graph.adj_matrix.A, dtype=self.floatx)
            self.x_tensor = gf.astensor(self.graph.node_attr, dtype=self.floatx)
            self.build(hids=hids)
            self.use_relu = use_relu
            self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)

            self.adj_changes = tf.Variable(tf.zeros_like(self.adj_tensor))
            self.x_changes = tf.Variable(tf.zeros_like(self.x_tensor))

        if reset:
            self.reset()
        return self

    def reset(self):
        super().reset()
        self.adj_flips = []
        self.nattr_flips = []

        with tf.device(self.device):
            self.adj_changes.assign(tf.zeros_like(self.adj_tensor))
            self.x_changes.assign(tf.zeros_like(self.x_tensor))
        return self

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

#     def log_likelihood_constraint(self, adj, modified_adj, ll_cutoff):
#         """
#         Computes a mask for entries that, if the edge corresponding to the entry is added/removed, would lead to the
#         log likelihood constraint to be violated.

#         """
#         t_d_min = tf.constant(2., dtype=self.floatx)
#         t_possible_edges = tf.constant(np.array(
#             np.triu(np.ones([self.num_nodes, self.num_nodes]),
#                     k=1).nonzero()).T,
#             dtype=self.intx)
#         allowed_mask, current_ratio = likelihood_ratio_filter(
#             t_possible_edges, modified_adj, adj, t_d_min, ll_cutoff)

#         return allowed_mask, current_ratio

    @tf.function
    def get_perturbed_adj(self, adj, adj_changes):
        adj_changes_square = adj_changes - tf.linalg.band_part(adj_changes, 0, 0)
        adj_changes_sym = adj_changes_square + tf.transpose(adj_changes_square)
        clipped_adj_changes = self.clip(adj_changes_sym)
        return adj + clipped_adj_changes

    @tf.function
    def get_perturbed_x(self, x, x_changes):
        return x + self.clip(x_changes)

    def forward(self, x, adj):
        h = x
        for w in self.weights[:-1]:
            h = adj @ h @ w
            if self.use_relu:
                h = relu(h)

        return adj @ h @ self.weights[-1]

    def structure_score(self,
                        modified_adj,
                        adj_grad,
                        ll_constraint=None,
                        ll_cutoff=None):
        adj_meta_grad = adj_grad * (-2. * modified_adj + 1.)
        # Make sure that the minimum entry is 0.
        adj_meta_grad -= tf.reduce_min(adj_meta_grad)

#         if not self.allow_singleton:
#             # Set entries to 0 that could lead to singleton nodes.
#             singleton_mask = self.filter_potential_singletons(modified_adj)
#             adj_meta_grad *= singleton_mask

        # if ll_constraint:
        #     allowed_mask, self.ll_ratio = self.log_likelihood_constraint(
        #         modified_adj, self.adj_tensor, ll_cutoff)
        #     adj_meta_grad = adj_meta_grad * allowed_mask

        return tf.reshape(adj_meta_grad, [-1])

    def feature_score(self, modified_nx, x_grad):
        x_meta_grad = x_grad * (-2. * modified_nx + 1.)
        x_meta_grad -= tf.reduce_min(x_meta_grad)
        return tf.reshape(x_meta_grad, [-1])

    def clip(self, matrix):
        clipped_matrix = tf.clip_by_value(matrix, -1., 1.)
        return clipped_matrix


@TensorFlow.register()
class Metattack(BaseMeta):
    def process(self,
                train_nodes,
                unlabeled_nodes,
                self_training_labels,
                hids=[16],
                lr=0.1,
                epochs=100,
                momentum=0.9,
                lambda_=0.,
                use_relu=True,
                reset=True):

        self.lr = lr
        self.epochs = epochs
        self.momentum = momentum
        self.lambda_ = lambda_

        if lambda_ not in (0., 0.5, 1.):
            raise ValueError(
                'Invalid value of `lambda_`, allowed values [0: (meta-self), 1: (meta-train), 0.5: (meta-both)].'
            )
        return super().process(train_nodes=train_nodes,
                               unlabeled_nodes=unlabeled_nodes,
                               self_training_labels=self_training_labels,
                               hids=hids,
                               use_relu=use_relu,
                               reset=reset)

    def build(self, hids):
        hids = gf.repeat(hids)
        weights, w_velocities = [], []
        zeros_initializer = zeros()

        pre_hid = self.num_attrs
        for hid in hids + [self.num_classes]:
            shape = (pre_hid, hid)
            # use zeros_initializer temporary to save time
            weight = tf.Variable(zeros_initializer(shape=shape, dtype=self.floatx))
            w_velocity = tf.Variable(zeros_initializer(shape=shape, dtype=self.floatx))

            weights.append(weight)
            w_velocities.append(w_velocity)

            pre_hid = hid

        self.weights, self.w_velocities = weights, w_velocities

    def initialize(self):
        w_initializer = glorot_uniform()
        zeros_initializer = zeros()

        for w, wv in zip(self.weights, self.w_velocities):
            w.assign(w_initializer(w.shape, dtype=self.floatx))
            wv.assign(zeros_initializer(wv.shape, dtype=self.floatx))

    @tf.function
    def train_step(self, x, adj, index, labels):
        with tf.GradientTape() as tape:
            output = self.forward(x, adj)
            logit = tf.gather(output, index)
            loss = self.loss_fn(labels, logit)

        weight_grads = tape.gradient(loss, self.weights)
        return weight_grads

    def inner_train(self, x, adj):

        self.initialize()
        adj_norm = gf.normalize_adj_tensor(adj)

        for it in range(self.epochs):
            weight_grads = self.train_step(x, adj_norm, self.train_nodes, self.labels_train)

            for v, g in zip(self.w_velocities, weight_grads):
                v.assign(self.momentum * v + g)

            for w, v in zip(self.weights, self.w_velocities):
                w.assign_sub(self.lr * v)

    @tf.function
    def meta_grad(self):

        modified_adj, modified_nx = self.adj_tensor, self.x_tensor
        adj_tensor, x_tensor = self.adj_tensor, self.x_tensor
        persistent = self.structure_attack and self.feature_attack

        with tf.GradientTape(persistent=persistent) as tape:
            if self.structure_attack:
                modified_adj = self.get_perturbed_adj(adj_tensor, self.adj_changes)

            if self.feature_attack:
                modified_nx = self.get_perturbed_x(x_tensor, self.x_changes)

            adj_norm = gf.normalize_adj_tensor(modified_adj)
            output = self.forward(modified_nx, adj_norm)
            logit_labeled = tf.gather(output, self.train_nodes)
            logit_unlabeled = tf.gather(output, self.unlabeled_nodes)

            loss_labeled = self.loss_fn(self.labels_train, logit_labeled)
            loss_unlabeled = self.loss_fn(self.self_training_labels, logit_unlabeled)

            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        adj_grad, x_grad = None, None

        if self.feature_attack:
            x_grad = tape.gradient(attack_loss, self.x_changes)

        if self.structure_attack:
            adj_grad = tape.gradient(attack_loss, self.adj_changes)

        return x_grad, adj_grad

    def attack(self,
               num_budgets=0.05,
               structure_attack=True,
               feature_attack=False,
               ll_constraint=False,
               ll_cutoff=0.004,
               disable=False):
        super().attack(num_budgets, structure_attack, feature_attack)

        if ll_constraint:
            raise NotImplementedError(
                "`log_likelihood_constraint` has not been well tested."
                " Please set `ll_constraint=False` to achieve a better performance."
            )

        if feature_attack and not self.graph.is_binary():
            raise ValueError(
                "Attacks on the node features are currently only supported for binary attributes."
            )

        with tf.device(self.device):
            modified_adj, modified_nx = self.adj_tensor, self.x_tensor
            adj_tensor, x_tensor = self.adj_tensor, self.x_tensor
            adj_changes, x_changes = self.adj_changes, self.x_changes
            adj_flips, nattr_flips = self.adj_flips, self.nattr_flips

            self.inner_train(modified_nx, modified_adj)
            
            for it in tqdm(range(self.num_budgets),
                           desc='Peturbing Graph',
                           disable=disable):

                if structure_attack:
                    modified_adj = self.get_perturbed_adj(adj_tensor, adj_changes)

                if feature_attack:
                    modified_nx = self.get_perturbed_x(x_tensor,x_changes)

                self.inner_train(modified_nx, modified_adj)

                x_grad, adj_grad = self.meta_grad()

                adj_meta_score = tf.constant(0.0)
                x_meta_score = tf.constant(0.0)

                if structure_attack:
                    adj_meta_score = self.structure_score(modified_adj, adj_grad, ll_constraint, ll_cutoff)

                if feature_attack:
                    x_meta_score = self.feature_score(modified_nx, x_grad)

                if tf.reduce_max(adj_meta_score) >= tf.reduce_max(x_meta_score) and structure_attack:
                    adj_meta_argmax = tf.argmax(adj_meta_score)
                    row, col = divmod(adj_meta_argmax.numpy(), self.num_nodes)
                    adj_changes[row, col].assign(-2. * modified_adj[row, col] + 1.)
                    adj_changes[col, row].assign(-2. * modified_adj[col, row] + 1.)
                    adj_flips.append((row, col))
                elif tf.reduce_max(adj_meta_score) < tf.reduce_max(x_meta_score) and feature_attack:
                    x_meta_argmax = tf.argmax(x_meta_score)
                    row, col = divmod(x_meta_argmax.numpy(), self.num_attrs)
                    x_changes[row, col].assign(-2 * modified_nx[row, col] + 1)
                    nattr_flips.append((row, col))
                else:
                    warnings.warn(f"Do nothing at iter {it}. adj_meta_score={adj_meta_score}, x_meta_score={x_meta_score}",
                                  UserWarning)


@TensorFlow.register()
class MetaApprox(BaseMeta):
    def process(self,
                train_nodes,
                unlabeled_nodes,
                self_training_labels,
                hids=[16],
                lr=0.1,
                epochs=100,
                lambda_=0.,
                use_relu=True):

        self.lr = lr
        self.epochs = epochs
        self.lambda_ = lambda_

        if lambda_ not in (0., 0.5, 1.):
            raise ValueError(
                'Invalid value of `lambda_`, allowed values [0: (meta-self), 1: (meta-train), 0.5: (meta-both)].'
            )
        return super().process(train_nodes=train_nodes,
                               unlabeled_nodes=unlabeled_nodes,
                               self_training_labels=self_training_labels,
                               hids=hids,
                               use_relu=use_relu)

    def build(self, hids):
        hids = gf.repeat(hids)
        weights = []
        zeros_initializer = zeros()

        pre_hid = self.num_attrs
        for hid in hids + [self.num_classes]:
            shape = (pre_hid, hid)
            # use zeros_initializer temporary to save time
            weight = tf.Variable(zeros_initializer(shape=shape, dtype=self.floatx))
            weights.append(weight)
            pre_hid = hid

        self.weights = weights
        self.adj_grad_sum = tf.Variable(tf.zeros_like(self.adj_tensor))
        self.x_grad_sum = tf.Variable(tf.zeros_like(self.x_tensor))
        self.optimizer = Adam(self.lr, epsilon=1e-8)

    def initialize(self):

        w_initializer = glorot_uniform()
        zeros_initializer = zeros()

        for w in self.weights:
            w.assign(w_initializer(w.shape, dtype=self.floatx))

        if self.structure_attack:
            self.adj_grad_sum.assign(zeros_initializer(self.adj_grad_sum.shape, dtype=self.floatx))

        if self.feature_attack:
            self.x_grad_sum.assign(zeros_initializer(self.x_grad_sum.shape, dtype=self.floatx))

        # reset optimizer
        for var in self.optimizer.variables():
            var.assign(tf.zeros_like(var))

    @tf.function
    def meta_grad(self):
        self.initialize()

        modified_adj, modified_nx = self.adj_tensor, self.x_tensor
        adj_tensor, x_tensor = self.adj_tensor, self.x_tensor
        adj_grad_sum, x_grad_sum = self.adj_grad_sum, self.x_grad_sum
        optimizer = self.optimizer

        for it in tf.range(self.epochs):

            with tf.GradientTape(persistent=True) as tape:
                if self.structure_attack:
                    modified_adj = self.get_perturbed_adj(adj_tensor, self.adj_changes)

                if self.feature_attack:
                    modified_nx = self.get_perturbed_x(x_tensor, self.x_changes)

                adj_norm = gf.normalize_adj_tensor(modified_adj)
                output = self.forward(modified_nx, adj_norm)
                logit_labeled = tf.gather(output, self.train_nodes)
                logit_unlabeled = tf.gather(output, self.unlabeled_nodes)

                loss_labeled = self.loss_fn(self.labels_train, logit_labeled)
                loss_unlabeled = self.loss_fn(self.self_training_labels, logit_unlabeled)

                attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

            adj_grad, x_grad = None, None

            gradients = tape.gradient(loss_labeled, self.weights)
            optimizer.apply_gradients(zip(gradients, self.weights))

            if self.structure_attack:
                adj_grad = tape.gradient(attack_loss, self.adj_changes)
                adj_grad_sum.assign_add(adj_grad)

            if self.feature_attack:
                x_grad = tape.gradient(attack_loss, self.x_changes)
                x_grad_sum.assign_add(x_grad)

            del tape

        return x_grad_sum, adj_grad_sum

    def attack(self,
               num_budgets=0.05,
               structure_attack=True,
               feature_attack=False,
               ll_constraint=False,
               ll_cutoff=0.004,
               disable=False):

        super().attack(num_budgets, structure_attack, feature_attack)

        if ll_constraint:
            raise NotImplementedError(
                "`log_likelihood_constraint` has not been well tested."
                " Please set `ll_constraint=False` to achieve a better performance."
            )

        if feature_attack and not self.graph.is_binary():
            raise ValueError(
                "Attacks on the node features are currently only supported for binary attributes."
            )

        with tf.device(self.device):
            modified_adj, modified_nx = self.adj_tensor, self.x_tensor
            adj_tensor, x_tensor = self.adj_tensor, self.x_tensor
            adj_changes, x_changes = self.adj_changes, self.x_changes
            adj_flips, nattr_flips = self.adj_flips, self.nattr_flips

            for it in tqdm(range(self.num_budgets),
                           desc='Peturbing Graph',
                           disable=disable):

                x_grad, adj_grad = self.meta_grad()

                adj_meta_score = tf.constant(0.0)
                x_meta_score = tf.constant(0.0)

                if structure_attack:
                    modified_adj = self.get_perturbed_adj(adj_tensor, adj_changes)
                    adj_meta_score = self.structure_score(modified_adj, adj_grad, ll_constraint, ll_cutoff)

                if feature_attack:
                    modified_nx = self.get_perturbed_x(x_tensor, x_changes)
                    x_meta_score = self.feature_score(modified_nx, feature_grad)

                if tf.reduce_max(adj_meta_score) >= tf.reduce_max(x_meta_score) and structure_attack:
                    adj_meta_argmax = tf.argmax(adj_meta_score)
                    row, col = divmod(adj_meta_argmax.numpy(), self.num_nodes)
                    adj_changes[row, col].assign(-2. * modified_adj[row, col] + 1.)
                    adj_changes[col, row].assign(-2. * modified_adj[col, row] + 1.)
                    adj_flips.append((row, col))
                elif tf.reduce_max(adj_meta_score) < tf.reduce_max(x_meta_score) and feature_attack:
                    x_meta_argmax = tf.argmax(x_meta_score)
                    row, col = divmod(x_meta_argmax.numpy(), self.num_attrs)
                    x_changes[row, col].assign(-2 * modified_nx[row, col] + 1)
                    nattr_flips.append((row, col))
                else:
                    warnings.warn(f"Do nothing at iter {it}. adj_meta_score={adj_meta_score}, x_meta_score={x_meta_score}",
                                  UserWarning)
        return self
