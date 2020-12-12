import sys
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from typing import Union, Optional, List, Tuple, Any, Callable
from .base_graph import BaseGraph
from .graph import Graph
from .edge_graph import EdgeGraph
from .multi_graph import MultiGraph
from .multi_edge_graph import MultiEdgeGraph
from .apply import sparse_apply
from .io import load_npz, makedirs


class ListGraph(BaseGraph):
    """A list of Homogeneous or Heterogeneous graph."""
    multiple = None

    def __init__(self, *list_graphs,
                 metadata: Any = None,
                 copy: bool = True):
        collects = locals()
        collects.pop('self')
        super().__init__(**collects)

    @property
    def graphs(self):
        return self.list_graphs

    @property
    def g(self):
        return self.graphs

    @property
    def graph(self):
        return self.graphs[0] if self.graphs else None

    @ classmethod
    def from_npz(cls, filepath: str):
        assert not filepath.endswith(".npz"), filepath
        filepath = osp.abspath(osp.expanduser(filepath))
        name = osp.split(filepath)[1]
        graphs = []
        i = 0
        path = osp.join(filepath, f"{name}_{str(i)}.npz")
        while osp.exists(path):
            loader = load_npz(path)
            graph_cls = loader.pop("__class__", "Graph")
            assert graph_cls in {"Graph", "MultiGraph", "EdgeGraph", "MultiEdgeGraph"}, graph_cls
            graphs.append(eval(graph_cls)(**loader, copy=False))
            i += 1
            path = osp.join(filepath, f"{name}_{str(i)}.npz")
        if i == 0:
            raise RuntimeError("no files found!")
        print(f"All the graphs are loaded from {filepath} (identified from 0 to {i-1})", file=sys.stderr)
        return cls(*graphs, copy=True)

    def to_npz(self, filepath: str, apply_fn=sparse_apply):
        assert not filepath.endswith(".npz"), filepath
        filepath = osp.abspath(osp.expanduser(filepath))
        makedirs(filepath)
        name = osp.split(filepath)[1]
        for i, graph in enumerate(self.graphs):
            path = osp.join(filepath, f"{name}_{str(i)}")
            graph.to_npz(path)
        print(f"All the graphs are saved to {filepath} (identified from 0 to {i})", file=sys.stderr)
        return filepath

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        else:
            # it should be 'int' or 'numpy int'
            return self.graphs[key]

    def __len__(self):
        return len(self.graphs)

    def __iter__(self):
        yield from self.graphs

    def extra_repr(self):
        return f"graphs=({len(self)}), "
