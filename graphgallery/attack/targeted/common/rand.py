import random
import numpy as np
from graphgallery.utils import tqdm
from graphgallery.attack.targeted import Common
from ..targeted_attacker import TargetedAttacker


@Common.register()
class RAND(TargetedAttacker):
    def reset(self):
        super().reset()
        self.modified_degree = self.degree.copy()
        return self

    def process(self, reset=True):
        self.nodes_set = set(range(self.num_nodes))
        if reset:
            self.reset()
        return self

    def attack(self,
               target,
               num_budgets=None,
               threshold=0.5,
               direct_attack=True,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        super().attack(target, num_budgets, direct_attack, structure_attack,
                       feature_attack)

        if direct_attack:
            influence_nodes = [target]
        else:
            # influence_nodes = list(self.graph.neighbors(target))
            influence_nodes = self.graph.adj_matrix[target].indices.tolist()

        chosen = 0
        adj_flips = self.adj_flips

        with tqdm(total=self.num_budgets,
                  desc='Peturbing Graph',
                  disable=disable) as pbar:
            while chosen < self.num_budgets:

                # randomly choose to add or remove edges
                if np.random.rand() <= threshold:
                    delta = 1.0
                    edge = self.add_edge(influence_nodes)
                else:
                    delta = -1.0
                    edge = self.del_edge(influence_nodes)

                if edge is not None:
                    adj_flips[edge] = chosen
                    chosen += 1
                    u, v = edge
                    self.modified_degree[u] += delta
                    self.modified_degree[v] += delta
                    pbar.update(1)

        return self

    def add_edge(self, influence_nodes):
        u = random.choice(influence_nodes)
        neighbors = self.graph.adj_matrix[u].indices.tolist()
        potential_nodes = list(self.nodes_set - set(neighbors) -
                               set([self.target, u]))

        if len(potential_nodes) == 0:
            return None

        v = random.choice(potential_nodes)

        if not self.is_modified(u, v):
            return (u, v)
        else:
            return None

    def del_edge(self, influence_nodes):

        u = random.choice(influence_nodes)
        neighbors = self.graph.adj_matrix[u].indices.tolist()
        potential_nodes = list(set(neighbors) - set([self.target, u]))

        if len(potential_nodes) == 0:
            return None

        v = random.choice(potential_nodes)

        if not self.allow_singleton and (self.modified_degree[u] <= 1
                                         or self.modified_degree[v] <= 1):
            return None

        if not self.is_modified(u, v):
            return (u, v)
        else:
            return None
