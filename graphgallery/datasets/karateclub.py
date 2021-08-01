import os.path as osp
import pickle as pkl
from graphgallery import functional as gf

from ..data import Reader
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

    __node_level_url__ = "https://github.com/EdisonLeeeee/GraphData/raw/master/datasets/karateclub/node_level"
    __graph_level_url__ = "https://github.com/EdisonLeeeee/GraphData/raw/master/datasets/karateclub/graph_level"

    def __init__(self,
                 name,
                 root=None,
                 *,
                 transform=None,
                 verbose=True,
                 url=None,
                 remove_download=False):
        if not name in self.available_datasets():
            raise ValueError(
                f"Currently only support for these datasets {tuple(self.available_datasets().keys())}."
            )

        if name == 'reddit10k':
            self.level = "graph_level"
            self.__url__ = self.__graph_level_url__
        else:
            self.level = "node_level"
            self.__url__ = self.__node_level_url__

        super().__init__(name=name, root=root,
                         transform=transform,
                         verbose=verbose, url=url,
                         remove_download=remove_download)

    @staticmethod
    def available_datasets():
        return _DATASET

    @property
    def urls(self):
        return [f"{self.__url__}/{self.name}/{raw_filename}"
                for raw_filename in self.raw_filenames]

    def __process__(self):
        reader = Reader()
        filenames = self.raw_paths
        if self.level == "node_level":
            adj_matrix = reader.read_edges(filenames[0])
            node_attr = reader.read_csv_features(filenames[1])
            node_label = reader.read_target(filenames[2])
            graph = Graph(adj_matrix, node_attr, node_label, copy=False)
        else:
            adj_matrix = reader.read_graphs(filenames[0])
            graph_label = reader.read_target(filenames[1])
            graph = MultiGraph(adj_matrix, graph_label=graph_label, copy=False)

        cache = dict(graph=graph)
        return cache

    @property
    def download_dir(self):
        return osp.join(self.root, 'karateclub', self.level, self.name)

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
