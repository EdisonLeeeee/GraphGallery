import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import SparseCategoricalCrossentropy

import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.utils import tqdm
from graphgallery.attack.targeted import TensorFlow
from ..targeted_attacker import TargetedAttacker


@TensorFlow.register()
class IG(TargetedAttacker):
    # IG can conduct feature attack
    _allow_feature_attack = True

    def process(self, surrogate, reset=True):
        if isinstance(surrogate, gg.gallery.nodeclas.Trainer):
            surrogate = surrogate.model

        adj, x = self.graph.adj_matrix, self.graph.node_attr
        self.nodes_set = set(range(self.num_nodes))
        self.features_set = np.arange(self.num_attrs)

        with tf.device(self.device):
            self.surrogate = surrogate
            self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)
            self.x_tensor = gf.astensor(x)
            self.adj_tensor = gf.astensor(adj.A)
            self.adj_norm = gf.normalize_adj_tensor(self.adj_tensor)

        if reset:
            self.reset()
        return self

    def attack(self,
               target,
               num_budgets=None,
               direct_attack=True,
               structure_attack=True,
               feature_attack=False,
               steps=20,
               disable=False):

        super().attack(target, num_budgets, direct_attack, structure_attack,
                       feature_attack)

        if feature_attack and not self.graph.is_binary():
            raise RuntimeError(
                "Currently only attack binary node attributes are supported")

        if structure_attack:
            candidate_edges = self.get_candidate_edges()
            link_importance = self.get_link_importance(candidate_edges,
                                                       steps,
                                                       disable=disable)

        if feature_attack:
            candidate_features = self.get_candidate_features()
            feature_importance = self.get_feature_importance(
                candidate_features, steps, disable=disable)

        if structure_attack and not feature_attack:
            self.adj_flips = candidate_edges[gf.largest_indices(
                link_importance, self.num_budgets)[0]]
        elif feature_attack and not structure_attack:
            self.nattr_flips = candidate_features[gf.largest_indices(
                feature_importance, self.num_budgets)[0]]
        else:
            # both attacks are conducted
            link_selected = []
            feature_selected = []
            importance = np.hstack([link_importance, feature_importance])
            boundary = link_importance.size

            for index in gf.largest_indices(importance, self.num_budgets)[0]:
                if index < boundary:
                    link_selected.append(index)
                else:
                    feature_selected.append(index - boundary)

            if link_selected:
                self.adj_flips = candidate_edges[link_selected]

            if feature_selected:
                self.nattr_flips = candidate_features[feature_selected]
        return self

    def get_candidate_edges(self):
        num_nodes = self.num_nodes
        target = self.target
        adj = self.graph.adj_matrix

        if self.direct_attack:
            influence_nodes = [target]
            candidate_edges = np.column_stack(
                (np.tile(target,
                         num_nodes - 1), list(self.nodes_set - set([target]))))
        else:
            influence_nodes = adj[target].indices

            candidate_edges = np.row_stack([
                np.column_stack((np.tile(infl, num_nodes - 2),
                                 list(self.nodes_set - set([target, infl]))))
                for infl in influence_nodes
            ])

        if not self.allow_singleton:
            candidate_edges = gf.singleton_filter(candidate_edges, adj)

        return candidate_edges

    def get_candidate_features(self):
        num_attrs = self.num_attrs
        target = self.target
        adj = self.graph.adj_matrix

        if self.direct_attack:
            influence_nodes = [target]
            candidate_features = np.column_stack(
                (np.tile(target, num_attrs), self.features_set))
        else:
            influence_nodes = adj[target].indices
            candidate_features = np.row_stack([
                np.column_stack((np.tile(infl, num_attrs), self.features_set))
                for infl in influence_nodes
            ])

        return candidate_features

    def get_link_importance(self, candidates, steps, disable=False):

        adj = self.adj_tensor
        x = self.x_tensor
        mask = (candidates[:, 0], candidates[:, 1])
        target_index = gf.astensor([self.target])
        target_label = gf.astensor(self.target_label)
        baseline_add = adj.numpy()
        baseline_add[mask] = 1.0
        baseline_add = gf.astensor(baseline_add)
        baseline_remove = adj.numpy()
        baseline_remove[mask] = 0.0
        baseline_remove = gf.astensor(baseline_remove)
        edge_indicator = self.graph.adj_matrix[mask].A1 > 0

        edges = candidates[edge_indicator]
        non_edges = candidates[~edge_indicator]

        edge_gradients = tf.zeros(edges.shape[0])
        non_edge_gradients = tf.zeros(non_edges.shape[0])

        for alpha in tqdm(tf.linspace(0., 1.0, steps + 1),
                          desc='Computing link importance',
                          disable=disable):
            ###### Compute integrated gradients for removing edges ######
            adj_diff = adj - baseline_remove
            adj_step = baseline_remove + alpha * adj_diff

            gradients = self.compute_structure_gradients(
                adj_step, x, target_index, target_label)
            edge_gradients += -tf.gather_nd(gradients, edges)

            ###### Compute integrated gradients for adding edges ######
            adj_diff = baseline_add - adj
            adj_step = baseline_add - alpha * adj_diff

            gradients = self.compute_structure_gradients(
                adj_step, x, target_index, target_label)
            non_edge_gradients += tf.gather_nd(gradients, non_edges)

        integrated_grads = np.zeros(edge_indicator.size)
        integrated_grads[edge_indicator] = edge_gradients.numpy()
        integrated_grads[~edge_indicator] = non_edge_gradients.numpy()

        return integrated_grads

    def get_feature_importance(self, candidates, steps, disable=False):
        adj = self.adj_norm
        x = self.x_tensor
        mask = (candidates[:, 0], candidates[:, 1])
        target_index = gf.astensor([self.target])
        target_label = gf.astensor(self.target_label)
        baseline_add = x.numpy()
        baseline_add[mask] = 1.0
        baseline_add = gf.astensor(baseline_add)
        baseline_remove = x.numpy()
        baseline_remove[mask] = 0.0
        baseline_remove = gf.astensor(baseline_remove)
        feature_indicator = self.graph.node_attr[mask] > 0

        features = candidates[feature_indicator]
        non_features = candidates[~feature_indicator]

        feature_gradients = tf.zeros(features.shape[0])
        non_feature_gradients = tf.zeros(non_features.shape[0])

        for alpha in tqdm(tf.linspace(0., 1.0, steps + 1),
                          desc='Computing feature importance',
                          disable=disable):
            ###### Compute integrated gradients for removing features ######
            x_diff = x - baseline_remove
            x_step = baseline_remove + alpha * x_diff

            gradients = self.compute_feature_gradients(adj, x_step,
                                                       target_index,
                                                       target_label)
            feature_gradients += -tf.gather_nd(gradients, features)

            ###### Compute integrated gradients for adding features ######
            x_diff = baseline_add - x
            x_step = baseline_add - alpha * x_diff

            gradients = self.compute_feature_gradients(adj, x_step,
                                                       target_index,
                                                       target_label)
            non_feature_gradients += tf.gather_nd(gradients, non_features)

        integrated_grads = np.zeros(feature_indicator.size)
        integrated_grads[feature_indicator] = feature_gradients.numpy()
        integrated_grads[~feature_indicator] = non_feature_gradients.numpy()

        return integrated_grads

    @tf.function
    def compute_structure_gradients(self, adj, x, target_index, target_label):

        with tf.GradientTape() as tape:
            tape.watch(adj)
            adj_norm = gf.normalize_adj_tensor(adj)
            logit = self.surrogate([x, adj_norm])
            logit = tf.gather(logit, target_index)
            loss = self.loss_fn(target_label, logit)

        gradients = tape.gradient(loss, adj)
        return gradients

    @tf.function
    def compute_feature_gradients(self, adj, x, target_index, target_label):

        with tf.GradientTape() as tape:
            tape.watch(x)
            logit = self.surrogate([x, adj])
            logit = tf.gather(logit, target_index)
            loss = self.loss_fn(target_label, logit)

        gradients = tape.gradient(loss, x)
        return gradients
