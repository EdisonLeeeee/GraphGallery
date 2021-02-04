import sys
import glob
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


class DictGraph(BaseGraph):
    """A dict of Homogeneous or Heterogeneous graph."""
    multiple = None

    def __init__(self, metadata: Any = None,
                 copy: bool = True,
                 **dict_graphs):
        collects = locals()
        del collects['self']
        super().__init__(**collects)

    @property
    def graphs(self):
        return self.dict_graphs

    @property
    def g(self):
        return self.graphs

    @classmethod
    def from_npz(cls, filepath: str):
        assert not filepath.endswith(".npz"), filepath
        filepath = osp.abspath(osp.expanduser(filepath))
        graphs = {}
        paths = glob.glob(osp.join(filepath, "*.npz"))
        if not paths:
            raise RuntimeError("no files found!")

        for path in paths:
            loader = load_npz(path)
            graph_cls = loader.pop("__class__", "Graph")
            assert graph_cls in {"Graph", "MultiGraph", "EdgeGraph", "MultiEdgeGraph"}, graph_cls
            graph_name = osp.split(path)[1][:-4]
            graphs[graph_name] = eval(graph_cls)(**loader, copy=False)
        print(f"All the graphs are loaded from {filepath} (identified by its file name)", file=sys.stderr)
        return cls(**graphs, copy=True)

    def to_npz(self, filepath: str, apply_fn=sparse_apply):
        assert not filepath.endswith(".npz"), filepath
        filepath = osp.abspath(osp.expanduser(filepath))
        makedirs(filepath)
        for graph_name, graph in self.graphs.items():
            path = osp.join(filepath, f"{graph_name}")
            graph.to_npz(path)
        print(f"All the graphs are saved to {filepath} (identified by its name)", file=sys.stderr)
        return filepath

    def __len__(self):
        return len(self.graphs.keys()) if self.graphs else 0

    def __iter__(self):
        yield from self.graphs.items()

    def extra_repr(self):
        return f"graphs={tuple(self.graphs.keys())}, "
