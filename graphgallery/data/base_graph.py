from abc import ABC
from copy import copy as _copy, deepcopy as _deepcopy
from typing import Union, Tuple, List

# NxGraph = Union[nx.Graph, nx.DiGraph]
# Array1D = Union[List, np.ndarray]
# Matrix2D = Union[List[List], np.ndarray]
# LabelMatrix = Union[Array1D, Matrix2D]
# AdjMatrix = Union[sp.csr_matrix, sp.csc_matrix]


# MultiNxGraph = Union[List[NxGraph], Tuple[NxGraph]]
# MultiArray1D = Union[List[Array1D], Tuple[Array1D]]
# MultiMatrix2D = Union[List[Matrix2D], Tuple[Matrix2D]]
# MultiLabelMatrix = Union[List[LabelMatrix], Tuple[LabelMatrix]]
# MultiAdjMatrix = Union[List[AdjMatrix], Tuple[AdjMatrix]]


class BaseGraph(ABC):

    _adj_matrix = None
    _attr_matrix = None
    _labels = None
    metadata = None

    def __init__(self):
        ...

    @ property
    def n_nodes(self) -> int:
        ...

    @ property
    def n_edges(self) -> int:
        ...

    @ property
    def n_graphs(self) -> int:
        ...

    @ property
    def n_attrs(self) -> int:
        ...

    @ property
    def n_classes(self) -> int:
        ...

    @ property
    def A(self):
        """alias of adj_matrix."""
        return self.adj_matrix

    @ property
    def X(self):
        """alias of attr_matrix."""
        return self.attr_matrix

    @ property
    def Y(self):
        """alias of labels."""
        return self.labels

    @ property
    def D(self):
        """alias of degrees."""
        return self.degrees

    def __len__(self) -> int:
        return self.n_graphs

    def unpack(self) -> tuple:
        """Return the (A, X, Y) triplet."""
        return self.A, self.X, self.Y

    def raw(self) -> tuple:
        """Return the raw (A, X, Y) triplet."""
        return self._adj_matrix, self._attr_matrix, self._labels

    def is_labeled(self) -> bool:
        return self.labels is not None and len(self.labels) != 0

    def is_attributed(self) -> bool:
        return self.attr_matrix is not None and len(self.attr_matrix) != 0

    def copy(self, deepcopy: bool = False) -> "BaseGraph":
        cls = self.__class__
        if deepcopy:
            return _deepcopy(self)
        else:
            return _copy(self)

    def __copy__(self) -> "BaseGraph":
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> "BaseGraph":
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, _deepcopy(v, memo))
        return result
