from abc import ABC
from copy import copy as _copy, deepcopy as _deepcopy
from typing import Union, Tuple
from graphgallery.typing import SparseMatrix, MultiSparseMatrix, ArrayLike, MultiArrayLike, GraphType

class BaseGraph(ABC):

    def __init__(self):
        ...

    @property
    def n_nodes(self) -> int:
        ...

    @property
    def n_edges(self) -> int:
        ...

    @property
    def n_graphs(self) -> int:
        ...

    @property
    def n_attrs(self) -> int:
        ...

    @property
    def n_classes(self) -> int:
        ...

    @property
    def A(self) -> Union[SparseMatrix, MultiSparseMatrix]:
        """alias of adj_matrix."""
        return self.adj_matrix

    @property
    def X(self) -> Union[ArrayLike, MultiArrayLike]:
        """alias of attr_matrix."""
        return self.attr_matrix

    @property
    def Y(self) -> Union[ArrayLike, MultiArrayLike]:
        """alias of labels."""
        return self.labels

    @property
    def D(self) -> Union[Tuple[ArrayLike, ArrayLike], ArrayLike, MultiArrayLike]:
        """alias of degrees."""
        return self.degrees

    def __len__(self) -> int:
        return self.n_graphs

    def unpack(self) -> tuple:
        return self.A, self.X, self.Y
    
    def raw(self) -> tuple:
        """Return the raw (A, X, Y) triplet."""
        return self._adj_matrix, self._attr_matrix, self._labels

    def is_labeled(self) -> bool:
        return self.labels is not None

    def is_attributed(self) -> bool:
        return self.attr_matrix is not None

    def copy(self, deepcopy: bool=False) -> GraphType:
        cls = self.__class__
        if deepcopy:
            return _deepcopy(self)
        else:
            return _copy(self)

    def __copy__(self) -> GraphType:
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> GraphType:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, _deepcopy(v, memo))
        return result
