from graphgallery import functional as gf
from ..flip_attacker import FlipAttacker


class UntargetedAttacker(FlipAttacker):
    def reset(self):
        self.modified_adj = None
        self.modified_nx = None
        self.modified_degree = None
        self.num_budgets = None
        self.structure_attack = None
        self.feature_attack = None

        self.nattr_flips = gf.BunchDict()
        self.adj_flips = gf.BunchDict()
        self.is_reseted = True
        return self

    def attack(self, num_budgets, structure_attack, feature_attack):

        if not self.is_reseted:
            raise RuntimeError(
                'Before calling attack, you must reset your attacker. Use `attacker.reset()`.'
            )

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

        num_budgets = self.budget_as_int(num_budgets,
                                         max_perturbations=self.num_edges)

        self.num_budgets = num_budgets
        self.structure_attack = structure_attack
        self.feature_attack = feature_attack
        self.is_reseted = False

    def is_modified(self, u, v):
        if not isinstance(self.adj_flips, (dict, set)):
            warnings.warn(
                f'Time consuming to check if edge ({u}, {v}) in `adj_flips`, whose type is {type(self.adj_flips)}.',
                UserWarning,
            )
        return any((u == v, (u, v) in self.adj_flips, (v, u)
                    in self.adj_flips))
