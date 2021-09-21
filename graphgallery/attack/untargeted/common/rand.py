import random
import numpy as np
from graphgallery.utils import tqdm
from graphgallery.attack.untargeted import Common
from ..untargeted_attacker import UntargetedAttacker


@Common.register()
class RAND(UntargetedAttacker):
    def process(self, reset=True):
        self.nodes_set = set(range(self.num_nodes))
        if reset:
            self.reset()
        return self

    def reset(self):
        super().reset()
        self.modified_degree = self.degree.copy()
        return self

    def attack(self,
               num_budgets=0.05,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        super().attack(num_budgets, structure_attack, feature_attack)

        influence_nodes = list(self.nodes_set)
        adj_flips = self.adj_flips
        random_list = np.random.choice(2, self.num_budgets) * 2 - 1

        for remove_or_insert in tqdm(random_list,
                                     desc='Peturbing Graph',
                                     disable=disable):
            if remove_or_insert > 0:
                edge = self.add_edge(influence_nodes)
                while edge is None:
                    edge = self.add_edge(influence_nodes)

            else:
                edge = self.del_edge(influence_nodes)
                while edge is None:
                    edge = self.del_edge(influence_nodes)

            adj_flips[edge] = 1.0
            u, v = edge
            self.modified_degree[u] += remove_or_insert
            self.modified_degree[v] += remove_or_insert
        return self

    def add_edge(self, influence_nodes):
        u = random.choice(influence_nodes)
        # assume that the graph has not self-loops
        neighbors = self.graph.adj_matrix[u].indices.tolist()
        potential_nodes = list(self.nodes_set - set(neighbors))

        if len(potential_nodes) == 0:
            return None

        v = random.choice(potential_nodes)

        if not self.is_modified(u, v):
            return (u, v)
        else:
            return None

    def del_edge(self, influence_nodes):
        u = random.choice(influence_nodes)
        # assume that the graph has not self-loops
        neighbors = self.graph.adj_matrix[u].indices.tolist()
        potential_nodes = neighbors

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
