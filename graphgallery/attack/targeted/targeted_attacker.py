import numpy as np
import graphgallery as gg
from graphgallery import functional as gf
from ..flip_attacker import FlipAttacker


class TargetedAttacker(FlipAttacker):
    def process(self):
        raise NotImplementedError

    def reset(self):
        self.modified_adj = None
        self.modified_nx = None
        self.modified_degree = None
        self.target = None
        self.target_label = None
        self.num_budgets = None
        self.structure_attack = None
        self.feature_attack = None
        self.direct_attack = None

        self.nattr_flips = gf.BunchDict()
        self.adj_flips = gf.BunchDict()
        self.is_reseted = True
        return self

    def attack(self, target, num_budgets, direct_attack, structure_attack,
               feature_attack):

        if not self.is_reseted:
            raise RuntimeError(
                'Before calling attack, you must reset your attacker. Use `attacker.reset()`.'
            )

        if not gg.is_intscalar(target):
            raise ValueError(target)

        if not (structure_attack or feature_attack):
            raise RuntimeError(
                'Either `structure_attack` or `feature_attack` must be True.')

        if feature_attack and not self.allow_feature_attack:
            raise RuntimeError(
                f"{self.name} does NOT support attacking features."
                " If the model can conduct feature attack, please call `attacker.allow_feature_attack=True`."
            )

        if structure_attack and not self.allow_structure_attack:
            raise RuntimeError(
                f"{self.name} does NOT support attacking structures."
                " If the model can conduct structure attack, please call `attacker.allow_structure_attack=True`."
            )

        if num_budgets is None:
            num_budgets = int(self.graph.degree[target])
        else:
            num_budgets = self.budget_as_int(
                num_budgets, max_perturbations=self.graph.degree[target])

        self.target = target
        self.target_label = self.graph.node_label[
            target] if self.graph.node_label is not None else None
        self.num_budgets = num_budgets
        self.direct_attack = direct_attack
        self.structure_attack = structure_attack
        self.feature_attack = feature_attack
        self.is_reseted = False

    def is_modified(self, u, v):
        if self.direct_attack:
            return any((u == v, (u, v) in self.adj_flips, (v, u) in self.adj_flips))
        else:
            return any((u == v, self.target in (u, v), (u, v)
                        in self.adj_flips, (v, u) in self.adj_flips))
