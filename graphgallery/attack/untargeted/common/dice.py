import random
import numpy as np
from graphgallery.attack.untargeted import Common
from .rand import RAND


@Common.register()
class DICE(RAND):
    """For DICE
    if a graph with a few classes of nodes,
     it might get stuck in an endless loop
    """

    def add_edge(self, influence_nodes):
        u = random.choice(influence_nodes)
        neighbors = self.graph.adj_matrix[u].indices.tolist()
        potential_nodes = list(self.nodes_set - set(neighbors))

        if len(potential_nodes) == 0:
            return None

        v = random.choice(potential_nodes)

        label = self.graph.node_label

        if not self.is_modified(u, v) and (label[u] != label[v]):
            return (u, v)
        else:
            return None

    def del_edge(self, influence_nodes):

        u = random.choice(influence_nodes)
        neighbors = self.graph.adj_matrix[u].indices.tolist()
        potential_nodes = neighbors

        if len(potential_nodes) == 0:
            return None

        v = random.choice(potential_nodes)

        if not self.allow_singleton and (self.modified_degree[u] <= 1
                                         or self.modified_degree[v] <= 1):
            return None

        label = self.graph.node_label

        if not self.is_modified(u, v) and (label[u] == label[v]):
            return (u, v)
        else:
            return None
