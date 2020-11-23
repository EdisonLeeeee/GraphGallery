import sys
import collections
import numpy as np
import os.path as osp
from abc import ABC
from copy import copy as _copy, deepcopy as _deepcopy
from typing import Union, Tuple, List
from functools import lru_cache

from .collate import sparse_collate
from .utils import check

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

    def keys(self):
        # TODO: maybe using `tuple`?
        keys = {key for key in self.__dict__.keys() if self[key] is not None and not key.startswith("_")}
        return keys

    def items(self, collate_fn=None):
        if callable(collate_fn):
            return tuple(collate_fn(key, self[key]) for key in sorted(self.keys()))
        else:
            return tuple((key, self[key]) for key in sorted(self.keys()))

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

    def to_namedtuple(self):
        keys = self.keys()
        DataTuple = collections.namedtuple('DataTuple', keys)
        return DataTuple(*[self[key] for key in keys])

    def copy(self, deepcopy: bool = False):
        if deepcopy:
            return _deepcopy(self)
        else:
            return _copy(self)

    def update(self, **collections):
        # TODO: check the acceptable args
        copy = collections.pop('copy', False)
        collections = check(collections, copy=copy)
        for k, v in collections.items():
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

    def __repr__(self):
        excluded = {"metadata", "graph_labels"}

        string = f"{self.__class__.__name__}("
        for k, v in self.items():
            if v is None or k in excluded:
                continue
            string += f"{k}{getattr(v, 'shape', f'({v})')}, "
        string += f"graph_labels={(self['graph_labels'])})"
        return string
