import os
import json
import os.path as osp
import numpy as np
import scipy.sparse as sp
import pickle as pkl

from graphgallery import functional as gf

from ..data.io import read_csv
from ..data.graph import Graph
from ..data.multi_graph import MultiGraph
from .in_memory_dataset import InMemoryDataset

_DATASET = gf.BunchDict(deezer="deezer dataset (node-level)",
                        facebook="facebook dataset (node-level)",
                        github="github dataset (node-level)",
                        lastfm="lastfm dataset (node-level)",
                        twitch="twitch dataset (node-level)",
                        wikipedia="wikipedia dataset (node-level)",
                        reddit10k="reddit10k dataset (graph-level)",
                        )


class KarateClub(InMemoryDataset):
    """Datasets from `Karate Club: An API Oriented Open-source Python Framework for Unsupervised Learning on Graphs`, CIKM 2020
    <https://github.com/benedekrozemberczki/karateclub>
    """

    _node_level_url = "https://github.com/benedekrozemberczki/karateclub/raw/master/dataset/node_level/"
    _graph_level_url = "https://github.com/benedekrozemberczki/karateclub/raw/master/dataset/graph_level/"

    def __init__(self,
                 name,
                 root=None,
                 *,
                 transform=None,
                 verbose=True,
                 url=None,
                 remove_download=False):

        if name == 'reddit10k':
            self.level = "graph_level"
            self._url = self._graph_level_url
        else:
            self.level = "node_level"
            self._url = self._node_level_url
        super().__init__(name=name, root=root,
                         transform=transform,
                         verbose=verbose, url=url,
                         remove_download=remove_download)

    @staticmethod
    def available_datasets():
        return _DATASET

    @property
    def urls(self):
        return [f"{self._url}/{self.name}/{raw_filename}"
                for raw_filename in self.raw_filenames]

    def _process(self):
        reader = Reader()
        filenames = self.raw_paths
        if self.level == "node_level":
            adj_matrix = reader.read_edges(filenames[0])
            node_attr = reader.read_features(filenames[1])
            node_label = reader.read_target(filenames[2])
            graph = Graph(adj_matrix, node_attr, node_label, copy=False)
        else:
            adj_matrix = reader.read_graphs(filenames[0])
            graph_label = reader.read_target(filenames[1])
            graph = MultiGraph(adj_matrix, graph_label=graph_label, copy=False)

        cache = dict(graph=graph)
        with open(self.process_path, 'wb') as f:
            pkl.dump(cache, f)
        return cache

    @property
    def download_dir(self):
        return osp.join(self.root, 'karateclub', self.level, self.name)

    @property
    def process_filename(self):
        return f'{self.name}.pkl'

    @property
    def raw_filenames(self):
        if self.level == "node_level":
            return ["edges.csv", "features.csv", "target.csv"]
        else:
            return ["graphs.json", "target.csv"]

    @property
    def raw_paths(self):
        return [
            osp.join(self.download_dir, raw_filename)
            for raw_filename in self.raw_filenames
        ]


class Reader:

    @staticmethod
    def read_graphs(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            graphs = json.load(f)
        graphs = [gf.edge_to_sparse_adj(graphs[str(i)]) for i in range(len(graphs))]
        return graphs

    @staticmethod
    def read_edges(filepath):
        data = read_csv(filepath)
        row = np.array(data['id_1'])
        col = np.array(data['id_2'])
        N = max(row.max(), col.max()) + 1
        graph = sp.csr_matrix((np.ones(row.shape[0], dtype=np.float32), (row, col)), shape=(N, N))
        return graph

    @staticmethod
    def read_features(filepath):
        data = read_csv(filepath)
        row = np.array(data["node_id"])
        col = np.array(data["feature_id"])
        values = np.array(data["value"])
        node_count = max(row) + 1
        feature_count = max(col) + 1
        shape = (node_count, feature_count)
        features = sp.csr_matrix((values, (row, col)), shape=shape)
        return features

    @staticmethod
    def read_target(filepath):
        data = read_csv(filepath)
        target = np.array(data["target"])
        return target
