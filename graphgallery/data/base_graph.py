from abc import ABC
from copy import copy as _copy, deepcopy as _deepcopy


class Basegraph(ABC):

    def __init__(self):
        ...

    @property
    def n_nodes(self):
        ...

    @property
    def n_edges(self):
        ...

    @property
    def n_graphs(self):
        ...

    @property
    def n_attrs(self):
        ...

    @property
    def n_classes(self):
        pass

    @property
    def A(self):
        """alias of adj_matrix"""
        return self.adj_matrix

    @property
    def X(self):
        """alias of attr_matrix"""
        return self.attr_matrix

    @property
    def Y(self):
        """alias of labels"""
        return self.labels

    @property
    def D(self):
        """alias of degrees"""
        return self.degrees

    def __len__(self):
        return self.n_graphs

    def unpack(self):
        return self.A, self.X, self.Y
    
    def raw(self):
        """Return the raw (A, X, Y) triplet."""
        return self._adj_matrix, self._attr_matrix, self._labels

    def is_labeled(self):
        return self.labels is not None

    def is_attributed(self):
        return self.attr_matrix is not None

    def copy(self, deepcopy=False):
        cls = self.__class__
        if deepcopy:
            return _deepcopy(self)
        else:
            return _copy(self)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, _deepcopy(v, memo))
        return result
