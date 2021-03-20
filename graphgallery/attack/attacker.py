import abc
import numpy as np
import graphgallery as gg


class Attacker(gg.gallery.Model):
    _max_perturbations = 0
    _allow_feature_attack = False
    _allow_structure_attack = True
    _allow_singleton = False

    def __init__(self, graph, device="cpu", seed=None, name=None, **kwargs):
        super().__init__(graph, device=device, seed=seed, name=name, **kwargs)

        self.is_reseted = False

        self.modified_adj = None  # adjacency matrix
        self.modified_nx = None  # node attribute matrix
        self.modified_ex = None  # edge attribute matrix
        self.modified_ny = None  # node label
        self.modified_ey = None  # edge label
        self.modified_gy = None  # graph label

        self.num_nodes = graph.num_nodes
        self.num_classes = graph.num_node_classes
        self.num_attrs = graph.num_node_attrs
        self.num_edges = graph.num_edges
        self.degree = graph.degree
        # TODO: more?

    def process(self, reset=True):
        if reset:
            self.reset()
        return self

    def reset(self):
        return self

    @property
    def g(self):
        graph = self.graph.copy()
        # TODO: edge attributes?
        updates = dict(adj_matrix=self.A, node_attr=self.nx)
        graph.update(**updates)
        return graph

    @abc.abstractmethod
    def attack(self):
        '''for attack model.'''
        raise NotImplementedError

    def budget_as_int(self, num_budgets, max_perturbations):
        max_perturbations = max(max_perturbations, self.max_perturbations)

        if not gg.is_scalar(num_budgets) or num_budgets <= 0:
            raise ValueError(
                f"'num_budgets' must be a postive integer scalar. but got '{num_budgets}'."
            )

        if num_budgets > max_perturbations:
            raise ValueError(
                f"'num_budgets' should be less than or equal the maximum allowed perturbations: {max_perturbations}."
                "if you want to use larget budgets, you could set 'attacker.set_max_perturbations(a_large_budgets)'."
            )

        if num_budgets < 1.:
            num_budgets = max_perturbations * num_budgets

        return int(num_budgets)

    @property
    def allow_singleton(self):
        return self._allow_singleton

    @allow_singleton.setter
    def allow_singleton(self, state):
        self._allow_singleton = state

    @property
    def allow_structure_attack(self):
        return self._allow_structure_attack

    @allow_structure_attack.setter
    def allow_structure_attack(self, state):
        self._allow_structure_attack = state

    @property
    def allow_feature_attack(self):
        return self._allow_feature_attack

    @allow_feature_attack.setter
    def allow_feature_attack(self, state):
        self._allow_feature_attack = state

    @property
    def max_perturbations(self):
        return self._max_perturbations

    def set_max_perturbations(self, max_perturbations=np.inf, verbose=True):
        assert gg.is_scalar(max_perturbations), max_perturbations
        self._max_perturbations = max_perturbations
        if verbose:
            print(f"Set maximum perturbations: {max_perturbations}")
