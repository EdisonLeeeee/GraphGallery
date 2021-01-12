import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy

import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.utils import tqdm
from graphgallery.attack.targeted import TensorFlow
from ..targeted_attacker import TargetedAttacker


@TensorFlow.register()
class FGSM(TargetedAttacker):
    def process(self, surrogate, reset=True):
        if isinstance(surrogate, gg.gallery.Trainer):
            surrogate = surrogate.model
        with tf.device(self.device):
            self.surrogate = surrogate
            self.loss_fn = sparse_categorical_crossentropy
            self.x_tensor = gf.astensor(self.graph.node_attr)
        if reset:
            self.reset()
        return self

    def reset(self):
        super().reset()
        self.modified_degree = self.degree.copy()

        with tf.device(self.device):
            modified_adj = tf.Variable(self.graph.adj_matrix.A,
                                       dtype=self.floatx)
            self.modified_adj = modified_adj
            self.adj_changes = tf.zeros_like(modified_adj)
        return self

    def attack(self,
               target,
               num_budgets=None,
               direct_attack=True,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        super().attack(target, num_budgets, direct_attack, structure_attack,
                       feature_attack)

        if not direct_attack:
            raise NotImplementedError(
                f'{self.name} does NOT support indirect attack.')

        target_index, target_label = gf.astensors([self.target],
                                                  self.target_label)
        adj_matrix = self.graph.adj_matrix

        for it in tqdm(range(self.num_budgets),
                       desc='Peturbing Graph',
                       disable=disable):
            with tf.device(self.device):
                gradients = self.compute_gradients(self.modified_adj,
                                                   self.adj_changes,
                                                   target_index, target_label)

                modified_row = tf.gather(self.modified_adj, target_index)
                gradients = (gradients *
                             (-2 * modified_row + 1)).numpy().ravel()

            sorted_index = np.argsort(-gradients)
            for index in sorted_index:
                u = target
                v = index % adj_matrix.shape[0]
                exist = adj_matrix[u, v]
                if exist and not self.allow_singleton and (
                        self.modified_degree[u] <= 1
                        or self.modified_degree[v] <= 1):
                    continue
                if not self.is_modified(u, v):
                    self.adj_flips[(u, v)] = it
                    self.flip_edge(u, v, exist)
                    break
        return self

    @tf.function
    def compute_gradients(self, modified_adj, adj_changes, target_index,
                          target_label):

        with tf.GradientTape() as tape:
            tape.watch(adj_changes)
            adj = modified_adj + adj_changes
            adj_norm = gf.normalize_adj_tensor(adj)
            logit = self.surrogate([self.x_tensor, adj_norm])
            logit = tf.gather(logit, target_index)
            loss = self.loss_fn(target_label, logit, from_logits=True)

        gradients = tape.gradient(loss, adj_changes)
        return gradients

    def flip_edge(self, u, v, exist):

        weight = 1. - exist
        delta_d = 2. * weight - 1.

        self.modified_adj[u, v].assign(weight)
        self.modified_adj[v, u].assign(weight)

        self.modified_degree[u] += delta_d
        self.modified_degree[v] += delta_d
