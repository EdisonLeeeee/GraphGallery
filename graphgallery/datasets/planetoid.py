import os
import os.path as osp
import numpy as np
import pickle as pkl

from typing import Optional, List, Tuple, Callable, Union

from .in_memory_dataset import InMemoryDataset
from ..data.io import makedirs, files_exist, download_file
from ..data.preprocess import process_planetoid_datasets
from ..data.graph import Graph

_DATASETS = {'citeseer', 'cora', 'pubmed'}
_DATASET_URL = "https://github.com/EdisonLeeeee/" + \
    "GraphData/raw/master/datasets/planetoid"
Transform = Union[List, Tuple, str, List, Tuple, Callable]


class Planetoid(InMemoryDataset):
    r"""The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    The original url is: <https://github.com/kimiyoung/planetoid/raw/master/data>
    """

    _url = _DATASET_URL

    def __init__(self,
                 name: str,
                 root: Optional[str] = None,
                 url: Optional[str] = None,
                 transform: Optional[Transform] = None,
                 verbose: bool = True):
        name = str(name)

        if not name in self.available_datasets():
            raise ValueError(
                f"Currently only support for these datasets {self.available_datasets()}."
            )

        super().__init__(name, root, url, transform, verbose)

    @staticmethod
    def available_datasets():
        return _DATASETS

    def _download(self):
        makedirs(self.download_dir)
        download_file(self.download_paths, self.urls)

    def _process(self) -> dict:

        adj_matrix, node_attr, node_label, train_nodes, val_nodes, test_nodes = process_planetoid_datasets(
            self.name, self.raw_paths)

        graph = Graph(adj_matrix, node_attr, node_label, copy=False)
        return dict(graph=graph,
                    train_nodes=train_nodes,
                    val_nodes=val_nodes,
                    test_nodes=test_nodes)

    def split_nodes(self,
                    train_size: float = None,
                    val_size: float = None,
                    test_size: float = None,
                    random_state: Optional[int] = None) -> dict:

        if not all((train_size, val_size, test_size)):
            self.splits.update(self.split_cache)
            return self.splits
        else:
            return super().split_nodes(train_size, val_size, test_size,
                                       random_state)

    @property
    def urls(self) -> List[str]:
        return [f"{self._url}/{raw_filename}"
                for raw_filename in self.raw_filenames]

    @ property
    def download_dir(self):
        return osp.join(self.root, 'planetoid')

    @ property
    def processed_path(self) -> str:
        return None

    @ property
    def raw_filenames(self) -> List[str]:
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return ['ind.{}.{}'.format(self.name.lower(), name) for name in names]

    @ property
    def raw_paths(self) -> List[str]:
        return [
            osp.join(self.download_dir, raw_filename)
            for raw_filename in self.raw_filenames
        ]

    def list_files(self):
        return self.raw_paths
