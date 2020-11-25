import os
import json
import os.path as osp
import numpy as np
import networkx as nx
import pickle as pkl

from itertools import product
from typing import Optional, List, Tuple, Callable, Union

from .in_memory_dataset import InMemoryDataset
from ..data.preprocess import process_planetoid_datasets
from ..data.multi_graph import MultiGraph
import graphgallery.functional as F

Transform = Union[List, Tuple, str, List, Tuple, Callable]


class PPI(InMemoryDataset):
    r"""The protein-protein interaction networks from the `"Predicting
    Multicellular Function through Multi-layer Tissue Networks"
    <https://arxiv.org/abs/1707.04638>`_ paper, containing positional gene
    sets, motif gene sets and immunological signatures as features (50 in
    total) and gene ontology sets as labels (121 in total).

    The original url is: <https://data.dgl.ai/dataset/ppi.zip>
    """

#     _url = "https://raw.githubusercontent.com/EdisonLeeeee/"
#     "GraphData/master/datasets/ppi/ppi.zip"
    _url = 'https://data.dgl.ai/dataset/ppi.zip'

    def __init__(self, root: Optional[str] = None,
                 transform: Optional[Transform] = None,
                 verbose: bool = True):
        name = "ppi"
        super().__init__(name, root, transform, verbose)

    def _process(self) -> None:

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

            G = F.nx_graph_to_sparse_adj(G)
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

        graph = MultiGraph(adj_matrices, node_attrs, node_labels, graph_label=graph_labels)
        cache['graph'] = graph
        with open(self.processed_path, 'wb') as f:
            pkl.dump(cache, f)
        return cache

    def split_graphs(self, train_size=None,
                     val_size=None,
                     test_size=None,
                     random_state: Optional[int] = None):
        loader = self.split_cache
        graph = self.graph
        self.splits.update(dict(train_graphs=graph[loader['train']],
                                val_graphs=graph[loader['valid']],
                                test_graphs=graph[loader['test']]))
        return self.splits

    @property
    def url(self) -> str:
        return self._url

    @property
    def processed_filename(self):
        return f'{self.name}.pkl'

    @property
    def raw_filenames(self) -> List[str]:
        splits = ['train', 'valid', 'test']
        files = ['feats.npy', 'graph_id.npy', 'graph.json', 'labels.npy']
        return ['{}_{}'.format(s, f) for s, f in product(splits, files)]

    @property
    def download_paths(self):
        return [osp.join(self.download_dir, self.name + '.zip')]

    @property
    def raw_paths(self) -> List[str]:
        return [osp.join(self.download_dir, raw_filename) for raw_filename in self.raw_filenames]
