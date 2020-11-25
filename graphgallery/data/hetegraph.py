import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from typing import Union, Optional, List, Tuple, Any

from .base_graph import BaseGraph
from .preprocess import largest_connected_components, create_subgraph


class HeteGraph(BaseGraph):
    """Heterogeneous graph stored in Numpy array form."""
    multiple = False

    def __init__(self, edge_index=None, edge_weight=None,
                 edge_attr=None, edge_label=None, *,
                 node_attr=None, node_label=None,
                 node_graph_label=None,
                 graph_attr=None,
                 graph_label=None,
                 mapping=None,
                 metadata: Any = None,
                 copy: bool = False):

        collects = locals()
        del collects['self']
        self.update(**collects)

    @ property
    def num_nodes(self) -> int:
        """Get the number of nodes in the graph."""
        return self.edge_index.max() + 1

    @property
    def num_edges(self):
        """Get the number of edges in the graph.
        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        return self.edge_index.shape[1]

    @ property
    def num_graphs(self) -> int:
        """Get the number of graphs."""
        return 1

    @ property
    def num_node_attrs(self) -> int:
        """Get the number of attribute dimensions of the nodes."""
        return get_num_node_attrs(self.node_attr)

    @ property
    def num_node_classes(self) -> int:
        """Get the number of classes node_label of the nodes."""
        return self.node_label.max() + 1

    @property
    def e(self):
        """alias of edge_index."""
        return self.edge_index

    @property
    def w(self):
        """alias of edge_weight."""
        return self.edge_weight

    @property
    def x(self):
        """alias of edge_attr."""
        return self.edge_attr

    @property
    def y(self):
        """alias of edge_label."""
        return self.edge_label

    def extra_repr(self):
        excluded = {"metadata", "mapping"}
        string = ""
        blank = ' ' * (len(self.__class__.__name__) + 1)
        for k, v in self.items():
            if v is None or k in excluded:
                continue
            string += f"{k}{getattr(v, 'shape')},\n{blank}"
        return string
