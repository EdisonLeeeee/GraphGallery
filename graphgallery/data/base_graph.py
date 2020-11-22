import collections

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

    def __init__(self):
        ...

    @property
    def num_nodes(self):
        ...

    @property
    def num_edges(self):
        ...

    @property
    def num_graphs(self):
        ...

    @property
    def num_node_attrs(self):
        ...

    @property
    def num_node_classes(self):
        ...

    @property
    def A(self):
        """alias of adj_matrix."""
        return self.adj_matrix

    @property
    def x(self):
        """alias of node_attr."""
        return self.node_attr

    @property
    def y(self):
        """alias of node_labels."""
        return self.node_labels

    @property
    def D(self):
        """alias of degrees."""
        return self.degrees

    @property
    def keys(self):
        keys = {key for key in self.__dict__.keys() if self[key] is not None and not key.startswith("_")}
        return keys

    @property
    def items(self):
        for key in sorted(self.keys):
            yield key, self[key]

    def is_labeled(self):
        return self.node_labels is not None and len(self.node_labels) != 0

    def is_attributed(self):
        return self.node_attr is not None and len(self.node_attr) != 0

    @classmethod
    def from_dict(cls, dictionary: dict):
        graph = cls(**dictionary)
        return graph

    def to_dict(self):
        return dict(self.items)

    def to_namedtuple(self):
        keys = self.keys
        DataTuple = collections.namedtuple('DataTuple', keys)
        return DataTuple(*[self[key] for key in keys])

    def copy(self, deepcopy: bool = False):
        cls = self.__class__
        if deepcopy:
            return _deepcopy(self)
        else:
            return _copy(self)

    def update(self, dictionary: dict):
        for k, v in dictionary.items():
            self[k] = v

    def __len__(self):
        return self.num_graphs

    def __contains__(self, key):
        return key in self.keys

    def __call__(self, *keys):
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, _deepcopy(v, memo))
        return result

    def __repr__(self):
        string = f"{self.__class__.__name__}("

        updated = False
        for k, v in self.items:
            if v is None or k == 'metadata':
                continue
            string += f"{k}{getattr(v, 'shape', '(None)')}, "
            updated = True

        if updated:
            string = string[:-2]

        string += ")"
        return string
