import sys
import numpy as np


def generate_node_attr(graph, fill_weight=1.0, num_nodes=None):
    # TODO: multiple graph
    if not num_nodes:
        num_nodes = graph.num_nodes
    if graph.node_attr is None:
        graph.update({"node_attr": np.eye(num_nodes, dtype=np.float32) * fill_weight})
    else:
        print("Node attribute matrix exists. Default to it.", file=sys.stderr)
    return graph
