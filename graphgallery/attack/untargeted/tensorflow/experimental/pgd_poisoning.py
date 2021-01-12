import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import softmax

import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.attack.untargeted import TensorFlow
from ..pgd import PGD, MinMax


@TensorFlow.register()
class PGDPoisoning(PGD):
    def attack(self,
               num_budgets=0.05,
               sample_epochs=20,
               C=0.1,
               CW_loss=False,
               epochs=200,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        return super().attack(num_budgets=num_budgets,
                              sample_epochs=sample_epochs,
                              C=C,
                              CW_loss=CW_loss,
                              epochs=epochs,
                              structure_attack=structure_attack,
                              feature_attack=feature_attack,
                              disable=disable)

    @tf.function
    def compute_loss(self, victim_nodes):
        adj = self.get_perturbed_adj()
        adj_norm = gf.normalize_adj_tensor(adj)
        logit = self.surrogate([self.x_tensor, adj_norm])
        logit = softmax(tf.gather(logit, victim_nodes))

        if self.CW_loss:
            best_wrong_class = tf.argmax(logit - self.label_matrix, axis=1)
            loss = self.loss_fn(self.victim_labels, logit) - self.loss_fn(best_wrong_class, logit)

        else:
            loss = self.loss_fn(self.victim_labels, logit)

        return tf.reduce_sum(loss)


@TensorFlow.register()
class MinMaxPoisoning(MinMax):
    def attack(self,
               num_budgets=0.05,
               sample_epochs=20,
               C=0.1,
               CW_loss=False,
               epochs=200,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        return super().attack(num_budgets=num_budgets,
                              sample_epochs=sample_epochs,
                              C=C,
                              CW_loss=CW_loss,
                              epochs=epochs,
                              structure_attack=structure_attack,
                              feature_attack=feature_attack,
                              disable=disable)

    @tf.function
    def compute_loss(self, victim_nodes):
        adj = self.get_perturbed_adj()
        adj_norm = gf.normalize_adj_tensor(adj)
        logit = self.surrogate([self.x_tensor, adj_norm])
        logit = softmax(tf.gather(logit, victim_nodes))

        if self.CW_loss:
            best_wrong_class = tf.argmax(logit - self.label_matrix, axis=1)
            loss = self.loss_fn(self.victim_labels, logit) - self.loss_fn(best_wrong_class, logit)
        else:
            loss = self.loss_fn(self.victim_labels, logit)

        return tf.reduce_sum(loss)
