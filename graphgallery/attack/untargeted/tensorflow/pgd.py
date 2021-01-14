import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.activations import softmax

import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.utils import tqdm
from graphgallery.attack.untargeted import TensorFlow
from ..untargeted_attacker import UntargetedAttacker


@TensorFlow.register()
class PGD(UntargetedAttacker):
    """PGD cannot ensure that there is not singleton after attack.
        https://github.com/KaidiXu/GCN_ADV_Train
    """

    def process(self,
                surrogate,
                train_nodes,
                unlabeled_nodes=None,
                reset=True):
        assert isinstance(surrogate, gg.gallery.DenseGCN), surrogate

        # poisoning attack in DeepRobust
        if unlabeled_nodes is None:
            victim_nodes = gf.asarray(train_nodes)
            victim_labels = self.graph.node_label[victim_nodes]
        else:  # Evasion attack in original paper
            self_training_labels = self.estimate_self_training_labels(surrogate, unlabeled_nodes)
            victim_nodes = np.hstack([train_nodes, unlabeled_nodes])
            victim_labels = np.hstack([self.graph.node_label[train_nodes], self_training_labels])

        with tf.device(self.device):
            adj_tensor = gf.astensor(self.graph.adj_matrix.A)
            self.victim_nodes = gf.astensor(victim_nodes)
            self.victim_labels = gf.astensor(victim_labels)
            self.adj_tensor = adj_tensor
            self.x_tensor = gf.astensor(self.graph.node_attr)
            self.complementary = tf.ones_like(adj_tensor) - tf.eye(self.num_nodes) - 2. * adj_tensor
            self.loss_fn = sparse_categorical_crossentropy
            self.adj_changes = tf.Variable(tf.zeros_like(adj_tensor))
            self.surrogate = surrogate.model

            # used for `CW_loss=True`
            self.label_matrix = tf.gather(tf.eye(self.num_classes), self.victim_labels)
            self.range_idx = tf.range(victim_nodes.size, dtype=self.intx)
            self.indices_real = tf.stack([self.range_idx, self.victim_labels],
                                         axis=1)
        if reset:
            self.reset()
        return self

    def reset(self):
        super().reset()
        with tf.device(self.device):
            self.adj_changes.assign(tf.zeros_like(self.adj_changes))
        return self

    def estimate_self_training_labels(self, surrogate, victim_nodes):
        self_training_labels = surrogate.predict(victim_nodes).argmax(1)
        return self_training_labels.astype(self.intx, copy=False)

    def attack(self,
               num_budgets=0.05,
               sample_epochs=20,
               C=None,
               CW_loss=False,
               epochs=100,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        super().attack(num_budgets, structure_attack, feature_attack)

        self.CW_loss = CW_loss
        if not C:
            if CW_loss:
                C = 0.1
            else:
                C = 200

        with tf.device(self.device):
            for epoch in tqdm(range(epochs),
                              desc='PGD Training',
                              disable=disable):
                gradients = self.compute_gradients(self.victim_nodes)
                lr = C / np.sqrt(epoch + 1)
                self.adj_changes.assign_add(lr * gradients)
                self.projection()

            best_s = self.random_sample(sample_epochs, disable=disable)
            self.adj_flips = np.transpose(np.where(best_s > 0.))
        return self

    @tf.function
    def compute_gradients(self, victim_nodes):
        with tf.GradientTape() as tape:
            tape.watch(self.adj_changes)
            loss = self.compute_loss(victim_nodes)

        gradients = tape.gradient(loss, self.adj_changes)
        return gradients

    @tf.function
    def compute_loss(self, victim_nodes):
        adj = self.get_perturbed_adj()
        adj_norm = gf.normalize_adj_tensor(adj)
        logit = self.surrogate([self.x_tensor, adj_norm])
        logit = tf.gather(logit, victim_nodes)
        logit = softmax(logit)
        
        if self.CW_loss:
            best_wrong_class = tf.argmax(logit - self.label_matrix, axis=1,
                                         output_type=self.intx)
            indices_attack = tf.stack([self.range_idx, best_wrong_class], axis=1)
            margin = tf.gather_nd(logit, indices_attack) - tf.gather_nd(logit, self.indices_real) - 0.2
            loss = tf.minimum(margin, 0.)
            return tf.reduce_sum(loss)
        else:
            loss = self.loss_fn(self.victim_labels, logit)

            return tf.reduce_mean(loss)

    @tf.function
    def get_perturbed_adj(self):
        adj_triu = tf.linalg.band_part(self.adj_changes, 0, -1) - tf.linalg.band_part(self.adj_changes, 0, 0)
        adj_changes = adj_triu + tf.transpose(adj_triu)
        adj = self.complementary * adj_changes + self.adj_tensor
        return adj

    def projection(self):
        clipped_matrix = self.clip(self.adj_changes)
        num_modified = tf.reduce_sum(clipped_matrix)

        if num_modified > self.num_budgets:
            left = tf.reduce_min(self.adj_changes - 1.)
            right = tf.reduce_max(self.adj_changes)
            miu = self.bisection(left, right, epsilon=1e-5)
            clipped_matrix = self.clip(self.adj_changes - miu)
        else:
            pass

        self.adj_changes.assign(clipped_matrix)

    def bisection(self, a, b, epsilon):
        def func(x):
            clipped_matrix = self.clip(self.adj_changes - x)
            return tf.reduce_sum(clipped_matrix) - self.num_budgets

        miu = a
        while (b - a) > epsilon:
            miu = (a + b) / 2
            # Check if middle point is root
            if func(miu) == 0:
                break
            # Decide the side to repeat the steps
            if func(miu) * func(a) < 0:
                b = miu
            else:
                a = miu
        return miu

    def clip(self, matrix):
        clipped_matrix = tf.clip_by_value(matrix, 0., 1.)
        return clipped_matrix

    def random_sample(self, sample_epochs=20, disable=False):
        best_loss = -10000
        best_s = None
        s = tf.linalg.band_part(self.adj_changes, 0, -1) - tf.linalg.band_part(self.adj_changes, 0, 0)
        for it in tqdm(range(sample_epochs),
                       desc='Random Sampling',
                       disable=disable):
            random_matrix = tf.random.uniform(shape=(self.num_nodes,
                                                     self.num_nodes),
                                              minval=0.,
                                              maxval=1.)
            sampled = tf.where(s > random_matrix, 1., 0.)
            if tf.reduce_sum(sampled) > self.num_budgets:
                continue

            with tf.device(self.device):
                self.adj_changes.assign(sampled)
                loss = self.compute_loss(self.victim_nodes)

            if best_loss < loss:
                best_loss = loss
                best_s = sampled

        assert best_s is not None, "Something wrong"
        return best_s.numpy()


@TensorFlow.register()
class MinMax(PGD):
    """MinMax cannot ensure that there is not singleton after attack."""

    def process(self,
                surrogate,
                train_nodes,
                unlabeled_nodes=None,
                lr=5e-3,
                reset=True):
        super().process(surrogate, train_nodes, unlabeled_nodes, reset=False)
        with tf.device(self.device):
            self.stored_weights = tf.identity_n(self.surrogate.weights)
            self.optimizer = Adam(lr)
        if reset:
            self.reset()
        return self

    def reset(self):
        super().reset()
        weights = self.surrogate.weights

        # restore surrogate weights
        for w1, w2 in zip(weights, self.stored_weights):
            w1.assign(w2)

        # reset optimizer
        for var in self.optimizer.variables():
            var.assign(tf.zeros_like(var))
        return self

    def attack(self,
               num_budgets=0.05,
               sample_epochs=20,
               C=None,
               CW_loss=False,
               epochs=100,
               update_per_epoch=20,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        super(PGD, self).attack(num_budgets, structure_attack, feature_attack)

        self.CW_loss = CW_loss

        if not C:
            if CW_loss:
                C = 0.1
            else:
                C = 200

        with tf.device(self.device):

            for epoch in tqdm(range(epochs),
                              desc='MinMax Training',
                              disable=disable):
                if (epoch + 1) % update_per_epoch == 0:
                    self.update_surrogate(self.victim_nodes)
                gradients = self.compute_gradients(self.victim_nodes)
                lr = C / np.sqrt(epoch + 1)
                self.adj_changes.assign_add(lr * gradients)
                self.projection()

            best_s = self.random_sample(sample_epochs)
            self.adj_flips = np.transpose(np.where(best_s > 0.))
        return self

    @tf.function
    def update_surrogate(self, victim_nodes):
        trainable_variables = self.surrogate.trainable_variables
        with tf.GradientTape() as tape:
            adj = self.get_perturbed_adj()
            loss = self.compute_loss(victim_nodes)

        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
