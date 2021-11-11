from ..base_transforms import GraphTransform
from ..transform import Transform
from ..network import largest_connected_components
from .subgraph import subgraph
__all__ = ['Standardize']


@Transform.register()
class Standardize(GraphTransform):
    def __call__(self, graph):
        # TODO: multiple graph
        assert not graph.is_multiple(), "NOT Supported for multiple graph"
        graph = graph.to_unweighted().to_undirected().remove_self_loop()
        nodes_to_keep = largest_connected_components(graph.adj_matrix)
        return subgraph(graph, nodes_to_keep=nodes_to_keep)
