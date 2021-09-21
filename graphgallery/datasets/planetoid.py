import os
import os.path as osp

from typing import Optional, List
from graphgallery import functional as gf

from .in_memory_dataset import InMemoryDataset
from ..data.io import makedirs, download_file
from ..data.preprocess import process_planetoid_datasets
from ..data.graph import Graph

_DATASETS = gf.BunchDict({"citeseer": "citeseer citation dataset",
                          "cora": "cora citation dataset",
                          "pubmed": "pubmed citation dataset",
                          "nell.0.1": "NELL dataset",
                          "nell.0.01": "NELL dataset",
                          "nell.0.001": "NELL dataset", })

_DATASET_URL = "https://github.com/EdisonLeeeee/" + \
    "GraphData/raw/master/datasets/planetoid"


class Planetoid(InMemoryDataset):
    r"""The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    The original url is: <https://github.com/kimiyoung/planetoid/raw/master/data>
    """

    __url__ = _DATASET_URL

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
        super().__init__(name=name, root=root,
                         transform=transform,
                         verbose=verbose, url=url,
                         remove_download=remove_download)

    @staticmethod
    def available_datasets():
        return _DATASETS

    def __process__(self):

        adj_matrix, node_attr, node_label, train_nodes, val_nodes, test_nodes = process_planetoid_datasets(
            self.name, self.raw_paths)

        graph = Graph(adj_matrix, node_attr, node_label, copy=False)
        return dict(graph=graph,
                    train_nodes=train_nodes,
                    val_nodes=val_nodes,
                    test_nodes=test_nodes)

    def split_nodes(self, *,
                    train: float = 0.1,
                    test: float = 0.8,
                    val: float = 0.1,
                    fixed_splits=True,
                    random_state: Optional[int] = None) -> dict:
        if fixed_splits:
            self.splits.update(self.split_cache)
            return self.splits
        else:
            return super().split_nodes(train=train, val=val, test=test,
                                       random_state=random_state)

    @property
    def urls(self):
        return [f"{self.__url__}/{raw_filename}"
                for raw_filename in self.raw_filenames]

    @property
    def download_dir(self):
        return osp.join(self.root, 'planetoid')

    @property
    def raw_filenames(self):
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return ['ind.{}.{}'.format(self.name.lower(), name) for name in names]

    @property
    def raw_paths(self):
        return [
            osp.join(self.download_dir, raw_filename)
            for raw_filename in self.raw_filenames
        ]

    def list_files(self):
        return self.raw_paths
