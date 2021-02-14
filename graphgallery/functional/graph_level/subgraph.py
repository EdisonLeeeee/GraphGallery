import copy
import numpy as np
import scipy.sparse as sp
import graphgallery as gg


def subgraph(graph, *,
             nodes_to_keep=None,
             nodes_to_remove=None):
    r"""Create a graph with the specified subset of nodes.
    Exactly one of (nodes_to_remove, nodes_to_keep) should be provided, while the other stays None.
    Note that to avoid confusion, it is required to pass node indices as named Parameters to this function.

    Parameters
    ----------
    graph : Graph
        Input graph.
    nodes_to_keep : array-like of int
        Indices of nodes that have to be kept.
    nodes_to_remove : array-like of int
        Indices of nodes that have to removed.

    Returns
    -------
    graph : Graph
        Graph with specified nodes removed.
    """
    # TODO: multiple graph
    assert isinstance(graph, gg.data.Graph), type(graph)
    graph = graph.copy()
    # Check that Parameters are passed correctly
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError("Only one of nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None:
        if len(nodes_to_remove) == 0:
            return graph
        nodes_to_keep = np.setdiff1d(np.arange(graph.num_nodes), nodes_to_remove)
    elif nodes_to_keep is not None:
        nodes_to_keep = np.sort(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    adj_matrix, node_attr, node_label = graph('adj_matrix',
                                              'node_attr',
                                              'node_label')
    graph.adj_matrix = adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if node_attr is not None:
        graph.node_attr = node_attr[nodes_to_keep]
    if node_label is not None:
        graph.node_label = node_label[nodes_to_keep]

    # TODO: remove?
    metadata = copy.deepcopy(graph.metadata)
    if metadata is not None and 'node_names' in metadata:
        graph.metadata['node_names'] = metadata['node_names'][nodes_to_keep]

    return graph
