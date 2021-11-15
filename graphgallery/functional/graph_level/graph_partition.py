
try:
    import pymetis
except ImportError:
    pymetis = None

import warnings
import numpy as np

from ..base_transforms import GraphTransform
from ..transform import Transform


def metis_clustering(graph, num_clusters):
    """Partitioning graph using Metis"""
    _, parts = pymetis.part_graph(num_clusters, adjacency=graph.tolil().rows)
    return parts


def random_clustering(num_nodes, num_clusters):
    """Partitioning graph randomly"""
    parts = np.random.choice(num_clusters, size=num_nodes)
    return parts


def louvain_clustering(graph):
    """Partitioning graph using louvain"""
    import community
    parts = community.community_louvain.best_partition(graph.to_networkx())
    return list(parts.values())


@Transform.register()
class GraphPartition(GraphTransform):
    def __init__(self, num_clusters: int = None, metis: bool = True):
        super().__init__()
        self.collect(locals())

    def __call__(self, graph):
        return graph_partition(graph, num_clusters=self.num_clusters, metis=self.metis_partition)


# TODO: accept a Graph and output a MultiGraph
def graph_partition(graph, num_clusters: int = None, partition: str = 'metis'):
    assert partition in {'metis', 'random', 'louvain'}, " only one of {'metis', 'random', 'louvain'} is accepted as `partition` argument."

    adj_matrix = graph.adj_matrix
    attr_matrix = graph.attr_matrix
    if num_clusters is None and partition != 'louvain':
        num_clusters = graph.num_classes
    # partition graph
    if partition == 'metis':
        if pymetis is None:
            raise RuntimeError('`pymetis is not installed, please install `pymetis` via `pip install pymetis`. More detailes please refer to <https://github.com/inducer/pymetis>.')
        parts = metis_clustering(adj_matrix, num_clusters)
    elif partition == 'louvain':
        if num_clusters is not None:
            warnings.warn('louvain does not require `num_clusters` argument.')
        parts = louvain_clustering(graph)
        num_clusters = np.unique(parts).size
    else:
        parts = random_clustering(adj_matrix.shape[0], num_clusters)

    cluster_member = [[] for _ in range(num_clusters)]
    for node_index, part in enumerate(parts):
        cluster_member[part].append(node_index)

    mapper = {}

    batch_adj, batch_x = [], []
    for cluster in range(num_clusters):

        nodes = sorted(cluster_member[cluster])
        mapper.update({old_id: new_id for new_id, old_id in enumerate(nodes)})

        mini_adj = adj_matrix[nodes][:, nodes]
        mini_x = attr_matrix[nodes]

        batch_adj.append(mini_adj)
        batch_x.append(mini_x)

    return batch_adj, batch_x, cluster_member
