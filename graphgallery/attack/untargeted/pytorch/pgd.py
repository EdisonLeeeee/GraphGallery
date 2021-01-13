import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.utils import tqdm
from graphgallery.attack.untargeted import PyTorch
from graphgallery.attack.untargeted.untargeted_attacker import UntargetedAttacker


@PyTorch.register()
class PGD(UntargetedAttacker):
    """PGD cannot ensure that there is not singleton after attack.
        https://github.com/KaidiXu/GCN_ADV_Train
    """

    def process(self,
                surrogate,
                train_nodes,
                unlabeled_nodes=None,
                reset=True):
        assert isinstance(surrogate, gg.gallery.GCN), surrogate

        # poisoning attack in DeepRobust
        if unlabeled_nodes is None:
            victim_nodes = gf.asarray(train_nodes)
            victim_labels = self.graph.node_label[victim_nodes]
        else:  # Evasion attack in original paper
            self_training_labels = self.estimate_self_training_labels(surrogate, unlabeled_nodes)
            victim_nodes = np.hstack([train_nodes, unlabeled_nodes])
            victim_labels = np.hstack([self.graph.node_label[train_nodes], self_training_labels])

        adj_tensor = gf.astensor(self.graph.adj_matrix.A, device=self.device)
        self.victim_nodes = gf.astensor(victim_nodes, device=self.device)
        self.victim_labels = gf.astensor(victim_labels, device=self.device)
        self.adj_tensor = adj_tensor
        self.x_tensor = gf.astensor(self.graph.node_attr, device=self.device)
        self.complementary = (torch.ones_like(adj_tensor) - torch.eye(self.num_nodes).to(self.device) - 2. * adj_tensor)
        self.loss_fn = nn.CrossEntropyLoss()
        self.adj_changes = nn.Parameter(torch.zeros_like(self.adj_tensor))
        self.surrogate = surrogate.model
        self.surrogate.eval()

        # # used for `CW_loss=True`
        self.label_matrix = torch.eye(self.num_classes)[self.victim_labels].to(self.device)
        self.range_idx = torch.arange(victim_nodes.size).to(self.device)
        self.indices_real = torch.stack([self.range_idx, self.victim_labels])
        if reset:
            self.reset()
        return self

    def reset(self):
        super().reset()
        self.adj_changes.data.zero_()
        return self

    def estimate_self_training_labels(self, surrogate, victim_nodes):
        self_training_labels = surrogate.predict(victim_nodes).argmax(1)
        return self_training_labels.astype(self.intx, copy=False)

    def attack(self,
               num_budgets=0.05,
               sample_epochs=20,
               C=None,
               CW_loss=False,
               epochs=200,
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

        for epoch in tqdm(range(epochs),
                          desc='PGD Training',
                          disable=disable):
            gradients = self.compute_gradients(self.victim_nodes)
            lr = C / np.sqrt(epoch + 1)
            self.adj_changes.data.add_(lr * gradients)
            self.projection()

        best_s = self.random_sample(sample_epochs, disable=disable)
        self.adj_flips = np.transpose(np.where(best_s > 0.))
        return self

    def compute_gradients(self, victim_nodes):
        loss = self.compute_loss(victim_nodes)

        gradients = torch.autograd.grad(loss, self.adj_changes)
        return gradients[0]

    def compute_loss(self, victim_nodes):
        adj = self.get_perturbed_adj()
        adj_norm = gf.normalize_adj_tensor(adj)
        logit = self.surrogate(self.x_tensor, adj_norm)[victim_nodes]

        if self.CW_loss:
            logit = F.log_softmax(logit, dim=1)
            best_wrong_class = (logit - 1000 * self.label_matrix).argmax(1)
            indices_attack = torch.stack([self.range_idx, best_wrong_class])
            margin = logit[self.indices_real] - logit[indices_attack] + 0.2
            loss = -torch.clamp(margin, min=0.) 
            return loss.mean()
        else:
            loss = self.loss_fn(logit, self.victim_labels)

            return loss

    def get_perturbed_adj(self):
        adj_triu = torch.triu(self.adj_changes, diagonal=1)
        adj_changes = adj_triu + adj_triu.t()
        adj = self.complementary * adj_changes + self.adj_tensor
        return adj

    def projection(self):
        clipped_matrix = self.clip(self.adj_changes)
        num_modified = clipped_matrix.sum()

        if num_modified > self.num_budgets:
            left = (self.adj_changes - 1.).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, epsilon=1e-5)
            clipped_matrix = self.clip(self.adj_changes - miu)
        else:
            pass

        self.adj_changes.data.copy_(clipped_matrix)

    def bisection(self, a, b, epsilon):
        def func(x):
            clipped_matrix = self.clip(self.adj_changes - x)
            return clipped_matrix.sum() - self.num_budgets

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
        clipped_matrix = torch.clamp(matrix, 0., 1.)
        return clipped_matrix

    def random_sample(self, sample_epochs=20, disable=False):
        best_loss = -10000
        best_s = None
        s = torch.triu(self.adj_changes, diagonal=1)
        for it in tqdm(range(sample_epochs),
                       desc='Random Sampling',
                       disable=disable):
            random_matrix = torch.zeros_like(s).uniform_(0, 1)
            sampled = torch.where(s > random_matrix, 1., 0.)
            if sampled.sum() > self.num_budgets:
                continue

            self.adj_changes.data.copy_(sampled)
            loss = self.compute_loss(self.victim_nodes)

            if best_loss < loss:
                best_loss = loss
                best_s = sampled

        return best_s.detach().cpu().numpy()
