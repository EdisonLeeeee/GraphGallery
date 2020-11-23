import sys
import numpy as np
import os.path as osp
from abc import ABC
from copy import copy as _copy, deepcopy as _deepcopy
from typing import Union, Tuple, List
from functools import partial

from .collate import sparse_collate
from .utils import check_and_convert
from ..data_type import is_listlike

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
    multiple = None

    def __init__(self):
        ...

    @ property
    def num_nodes(self) -> int:
        """Get the number of nodes in the graph."""
        return get_num_nodes(self.adj_matrix)

    @property
    def num_edges(self):
        """Get the number of edges in the graph.
        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        return get_num_edges(self.adj_matrix, self.is_directed())

    @ property
    def num_graphs(self) -> int:
        """Get the number of graphs."""
        return get_num_graphs(self.adj_matrix)

    @ property
    def num_node_attrs(self) -> int:
        """Get the number of attribute dimensions of the nodes."""
        return get_num_node_attrs(self.node_attr)

    @ property
    def num_node_classes(self) -> int:
        """Get the number of classes node_labels of the nodes."""
        return get_num_node_classes(self.node_labels)

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

    def keys(self):
        # TODO: maybe using `tuple`?
        keys = {key for key in self.__dict__.keys() if self[key] is not None and not key.startswith("_")}
        return keys

    def items(self, collate_fn=None):
        if callable(collate_fn):
            return tuple(collate_fn(key, self[key]) for key in sorted(self.keys()))
        else:
            return tuple((key, self[key]) for key in sorted(self.keys()))

    def dicts(self, collate_fn=None):
        return dict(self.items(collate_fn=collate_fn))

    def is_directed(self) -> bool:
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        adj_matrix = self.adj_matrix
        if isinstance(adj_matrix, (list, tuple)):
            adj_matrix = adj_matrix[0]
        return (adj_matrix != adj_matrix.T).sum() != 0

    def is_labeled(self):
        return self.node_labels is not None and len(self.node_labels) != 0

    def is_attributed(self):
        return self.node_attr is not None and len(self.node_attr) != 0

    @ classmethod
    def from_npz(cls, filepath: str):
        filepath = osp.abspath(osp.expanduser(osp.realpath(filepath)))

        if not filepath.endswith('.npz'):
            filepath = filepath + '.npz'

        if osp.isfile(filepath):
            with np.load(filepath, allow_pickle=True) as loader:
                loader = dict(loader)

                for k, v in loader.items():
                    if v.dtype.kind == 'O':
                        loader[k] = v.tolist()

                return cls(**loader)
        else:
            raise ValueError(f"{filepath} doesn't exist.")

    def to_npz(self, filepath: str, collate_fn=sparse_collate):

        filepath = osp.abspath(osp.expanduser(osp.realpath(filepath)))

        data_dict = {k: v for k, v in self.items(collate_fn=collate_fn) if v is not None}
        np.savez_compressed(filepath, **data_dict)
        print(f"Save to {filepath}.", file=sys.stderr)

        return filepath

    @ classmethod
    def from_dict(cls, dictionary: dict):
        graph = cls(**dictionary)
        return graph

    def to_dict(self):
        return dict(self.items())

    def copy(self, deepcopy: bool = False):
        if deepcopy:
            return _deepcopy(self)
        else:
            return _copy(self)

    def update(self, *, collate_fn=None, copy=False, **collates):
        if collate_fn is None:
            # TODO: check the acceptable args
            collate_fn = partial(check_and_convert,
                                 multiple=self.multiple,
                                 copy=copy)

        for k, v in collates.items():
            k, v = collate_fn(k, v)
            self[k] = v

    def __len__(self):
        return self.num_graphs

    def __contains__(self, key):
        assert isinstance(key, str)
        return key in self.keys()

    def __call__(self, *keys):
        for key in sorted(self.keys()) if not keys else keys:
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

    def extra_repr(self):
        return ""

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()}" \
            + f"metadata={tuple(self.metadata.keys()) if isinstance(self.metadata, dict) else self.metadata})"


def get_num_nodes(adj_matrices):
    if adj_matrices is None:
        return 0

    if is_listlike(adj_matrices):
        return sum(get_num_nodes(adj_matrix) for adj_matrix in adj_matrices)

    return adj_matrices.shape[0]


def get_num_graphs(adj_matrices):
    if adj_matrices is None:
        return 0

    if is_listlike(adj_matrices):
        # return sum(get_num_graphs(adj_matrix) for adj_matrix in adj_matrices)
        return len(adj_matrices)

    return 1


def get_num_edges(adj_matrices, is_directed=False):
    if adj_matrices is None:
        return 0

    if is_listlike(adj_matrices):
        return sum(get_num_edges(adj_matrix) for adj_matrix in adj_matrices)

    if is_directed:
        return int(adj_matrices.nnz)
    else:
        A = adj_matrices
        num_diag = (A.diagonal() != 0).sum()
        return int((A.nnz - num_diag) / 2) + int(num_diag)


def get_num_node_attrs(node_attrs):
    if node_attrs is None:
        return 0

    if is_listlike(node_attrs):
        return max(get_num_node_attrs(node_attr for node_attr in node_attrs))

    return node_attrs.shape[1]


def get_num_node_classes(node_labels):
    if node_labels is None:
        return 0

    if is_listlike(node_labels):
        return max(get_num_node_classes(node_label for node_label in node_labels))

    if node_labels.ndim == 1:
        return node_labels.max() + 1
    else:
        return node_labels.shape[1]
