import sys
import numpy as np
import os.path as osp
from functools import partial
from copy import copy as _copy, deepcopy as _deepcopy
from typing import Union, Tuple, List

from .collate import check_and_convert


class BaseGraph:
    multiple = None

    def __init__(self):
        # something needs to be done here?
        ...

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
    def num_node_classes(self) -> int:
        """Get the number of classes node_label of the nodes."""
        raise NotImplementedError

    def keys(self):
        # TODO: maybe using `tuple`?
        keys = {key for key in self.__dict__.keys() if self[key] is not None and not key.startswith("_")}
        return sorted(keys)

    def items(self, collate_fn=None):
        if callable(collate_fn):
            return tuple(collate_fn(key, self[key]) for key in self.keys())
        else:
            return tuple((key, self[key]) for key in self.keys())

    def dicts(self, collate_fn=None):
        return dict(self.items(collate_fn=collate_fn))

    @ classmethod
    def from_dict(cls, dictionary: dict):
        graph = cls(**dictionary)
        return graph

    def to_dict(self):
        return dict(self.items())

    def update(self, *, collate_fn=None, copy=False, **collects):
        if collate_fn is None:
            collate_fn = partial(check_and_convert,
                                 multiple=self.multiple,
                                 copy=copy)

        for k, v in collects.items():
            k, v = collate_fn(k, v)
            self[k] = v

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
            yield self[key]

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
