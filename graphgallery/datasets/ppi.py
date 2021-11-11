import os
import json
import os.path as osp
import numpy as np
import networkx as nx
import scipy.sparse as sp
import pickle as pkl

from itertools import product
from typing import Optional, List

from .in_memory_dataset import InMemoryDataset
from ..data.multi_graph import MultiGraph
from graphgallery import functional as gf


class PPI(InMemoryDataset):
    r"""The protein-protein interaction networks from the `"Predicting
    Multicellular Function through Multi-layer Tissue Networks"
    <https://arxiv.org/abs/1707.04638>`_ paper, containing positional gene
    sets, motif gene sets and immunological signatures as features (50 in
    total) and gene ontology sets as labels (121 in total).

    The original url is: <https://data.dgl.ai/dataset/ppi.zip>
    """

    #     __url__ = "https://raw.githubusercontent.com/EdisonLeeeee/"
    #     "GraphData/master/datasets/ppi/ppi.zip"
    __url__ = 'https://data.dgl.ai/dataset/ppi.zip'

    def __init__(self,
                 root=None,
                 *,
                 transform=None,
                 verbose=True,
                 url=None,
                 remove_download=True):

        super().__init__(name="ppi", root=root,
                         transform=transform,
                         verbose=verbose, url=url,
                         remove_download=remove_download)

    @staticmethod
    def available_datasets():
        return gf.BunchDict(ppi="ppi dataset")

    def __process__(self):

        adj_matrices = []
        node_attrs = []
        node_labels = []
        graph_labels = []
        path = self.download_dir
        cache = {}
        last = 0
        for split in ("train", "valid", "test"):
            idx = np.load(os.path.join(path, f"{split}_graph_id.npy"))
            x = np.load(os.path.join(path, f"{split}_feats.npy"))
            y = np.load(os.path.join(path, f"{split}_labels.npy"))
            nx_graph_path = os.path.join(path, f"{split}_graph.json")

            with open(nx_graph_path, "r", encoding="utf-8") as f:
                G = nx.DiGraph(nx.json_graph.node_link_graph(json.load(f)))

            G = nx_graph_to_sparse_adj(G)
            idx = idx - idx.min()
            for i in range(idx.max() + 1):
                mask = idx == i
                adj_matrices.append(G[mask][:, mask])
                node_attrs.append(x[mask])
                node_labels.append(y[mask])
                graph_labels.append(i)

            now = len(adj_matrices)
            cache[split] = slice(last, now)
            last = now

        graph = MultiGraph(adj_matrices,
                           node_attrs,
                           node_labels,
                           graph_label=graph_labels)
        cache['graph'] = graph
        with open(self.process_path, 'wb') as f:
            pkl.dump(cache, f)
        return cache

    def split_graphs(self,
                     train_size=None,
                     val_size=None,
                     test_size=None,
                     split_by=None,
                     random_state: Optional[int] = None):
        loader = self.split_cache
        graph = self.graph
        self.splits.update(
            dict(train_graphs=graph[loader['train']],
                 val_graphs=graph[loader['valid']],
                 test_graphs=graph[loader['test']]))
        return self.splits

    @property
    def process_filename(self):
        return f'{self.name}.pkl'

    @property
    def raw_filenames(self):
        splits = ['train', 'valid', 'test']
        files = ['feats.npy', 'graph_id.npy', 'graph.json', 'labels.npy']
        return ['{}_{}'.format(s, f) for s, f in product(splits, files)]

    @property
    def download_paths(self):
        return [osp.join(self.download_dir, self.name + '.zip')]

    @property
    def raw_paths(self):
        return [
            osp.join(self.download_dir, raw_filename)
            for raw_filename in self.raw_filenames
        ]


def nx_graph_to_sparse_adj(graph):
    num_nodes = graph.number_of_nodes()
    data = np.asarray(list(graph.edges().data('weight', default=1.0)))
    edge_index = data[:, :2].T.astype(np.int64)
    edge_weight = data[:, -1].T.astype(np.float32)
    adj_matrix = sp.csr_matrix((edge_weight, edge_index), shape=(num_nodes, num_nodes))
    return adj_matrix
