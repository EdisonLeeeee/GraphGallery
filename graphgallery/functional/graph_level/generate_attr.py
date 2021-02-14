import sys
import numpy as np

from ..base_transforms import GraphTransform
from ..transform import Transform

__all__ = ['GenerateNodeAttr', 'GenerateEdgeAttr']


@Transform.register()
class GenerateNodeAttr(GraphTransform):
    def __init__(self, N=None, fill_weight=1.0):
        super().__init__()
        self.collect(locals())

    def __call__(self, graph):
        # TODO: multiple graph
        assert not graph.multiple
        N = self.N
        if not N:
            N = graph.num_nodes
            self.N = N
        assert N, "There are no nodes in the graph."
        if graph.node_attr is None:
            graph.update(node_attr=np.eye(N, dtype=np.float32) * self.fill_weight)
        else:
            print("Node attribute matrix exists. Default to it.", file=sys.stderr)
        return graph


@Transform.register()
class GenerateEdgeAttr(GraphTransform):
    def __init__(self, N=None, fill_weight=1.0):
        super().__init__()
        self.collect(locals())

    def __call__(self, graph):
        # TODO: multiple graph
        assert not graph.multiple
        N = self.N
        if not N:
            N = graph.num_edges
            self.N = N
        assert N, "There are no edges in the graph."
        if graph.edge_attr is None:
            graph.update(edge_attr=np.eye(N, dtype=np.float32) * self.fill_weight)
        else:
            print("Edge attribute matrix exists. Default to it.", file=sys.stderr)
        return graph
