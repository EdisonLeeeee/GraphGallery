import sys
import numpy as np
import os.path as osp
from functools import partial
from copy import copy as _copy, deepcopy as _deepcopy
from typing import Union, Tuple, List

from .apply import check_and_convert, sparse_apply
from .io import load_npz


class BaseGraph:
    multiple = None

    def __init__(self, *args, **kwargs):
        kwargs.pop('__class__', None)
        self.update(**kwargs)

    @ property
    def num_nodes(self) -> int:
        """Get the number of nodes in the graph."""
        raise NotImplementedError

    @property
    def num_edges(self):
        """Get the number of edges in the graph.
        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        raise NotImplementedError

    @ property
    def num_graphs(self) -> int:
        """Get the number of graphs."""
        raise NotImplementedError

    @ property
    def num_node_attrs(self) -> int:
        """Get the number of attribute dimensions of the nodes."""
        raise NotImplementedError

    @ property
    def num_edge_attrs(self) -> int:
        """Get the number of attribute dimensions of the edges."""
        raise NotImplementedError

    @ property
    def num_graph_attrs(self) -> int:
        """Get the number of attribute dimensions of the graphs."""
        raise NotImplementedError

    @ property
    def num_node_classes(self) -> int:
        """Get the number of node classes."""
        raise NotImplementedError

    @ property
    def num_edge_classes(self) -> int:
        """Get the number of edge classes."""
        raise NotImplementedError

    @ property
    def num_graph_classes(self) -> int:
        """Get the number of graph classes."""
        raise NotImplementedError

    def is_node_attributed(self) -> bool:
        return self.node_attr is not None

    def is_edge_attributed(self) -> bool:
        return self.edge_attr is not None

    def is_graph_attributed(self) -> bool:
        return self.graph_attr is not None

    def is_node_labeled(self) -> bool:
        return self.node_label is not None

    def is_edge_labeled(self) -> bool:
        return self.edge_label is not None

    def is_graph_labeled(self) -> bool:
        return self.graph_label is not None

    def keys(self):
        # maybe using `tuple`?
        keys = {key for key in self.__dict__.keys() if getattr(self, key, None) and not key.startswith("_")}
        return sorted(keys)

    def items(self, apply_fn=None):
        if callable(apply_fn):
            return tuple(apply_fn(key, getattr(self, key, None)) for key in self.keys())
        else:
            return tuple((key, getattr(self, key, None)) for key in self.keys())

    def dicts(self, apply_fn=None):
        return dict(self.items(apply_fn=apply_fn))

    @ classmethod
    def from_dict(cls, dictionary: dict):
        graph = cls(**dictionary)
        return graph

    def to_dict(self):
        return dict(self.items())

    @ classmethod
    def from_npz(cls, filepath: str):
        loader = load_npz(filepath)
        return cls(copy=False, **loader)

    def to_npz(self, filepath: str, apply_fn=sparse_apply):

        filepath = osp.abspath(osp.expanduser(osp.realpath(filepath)))

        data_dict = {k: v for k, v in self.items(apply_fn=apply_fn) if v is not None}
        np.savez_compressed(filepath, **data_dict)
        print(f"Save to {filepath}.", file=sys.stderr)

        return filepath

    def update(self, *, apply_fn=None, copy=False, **collects):
        if apply_fn is None:
            apply_fn = partial(check_and_convert,
                               multiple=self.multiple,
                               copy=copy)
        else:
            assert callable(apply_fn)

        for k, v in collects.items():
            k, v = apply_fn(k, v)
            setattr(self, k, v)

    def copy(self, deepcopy: bool = False):
        if deepcopy:
            return _deepcopy(self)
        else:
            return _copy(self)

    def __len__(self):
        return self.num_graphs

    def __contains__(self, key):
        assert isinstance(key, str)
        return key in self.keys()

    def __call__(self, *keys):
        for key in keys:
            yield getattr(self, key, None)

    def __dir__(self):
        return self.keys()

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
        return f"{self.__class__.__name__}({self.extra_repr()}" \
            + f"metadata={tuple(self.metadata.keys()) if isinstance(self.metadata, dict) else self.metadata})"

    def extra_repr(self):
        return ""
