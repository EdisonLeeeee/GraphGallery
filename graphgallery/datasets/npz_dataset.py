import os
import sys
import zipfile
import os.path as osp
import numpy as np

from typing import Optional, List, Tuple, Union, Callable
from .in_memory_dataset import InMemoryDataset
from ..data.io import makedirs, files_exist, download_file
from ..data.graph import Graph

_DATASETS = {
    'citeseer', 'citeseer_full', 'cora', 'cora_ml', 'cora_full', 'amazon_cs',
    'amazon_photo', 'coauthor_cs', 'coauthor_phy', 'polblogs', 'karate_club',
    'pubmed', 'flickr', 'blogcatalog', 'dblp', 'acm', 'uai'
}

_DATASET_URL = "https://github.com/EdisonLeeeee/GraphData/raw/master/datasets"


class NPZDataset(InMemoryDataset):

    _url = _DATASET_URL

    def __init__(self,
                 name: str,
                 root: Optional[str] = None,
                 url: Optional[str] = None,
                 transform=None,
                 verbose: bool = True):

        name = str(name)
        if not name in self.available_datasets():
            print(
                f"Dataset not found in available datasets. Using custom dataset: {name}.",
                file=sys.stderr)
            self.custom = True
        else:
            self.custom = False

        super().__init__(name, root, url, transform, verbose)

    @staticmethod
    def available_datasets():
        return _DATASETS

    def _download(self) -> None:
        if not self.custom:
            makedirs(self.download_dir)
            download_file(self.download_paths, self.urls)

        elif not osp.exists(self.raw_paths[0]):
            raise RuntimeError(
                f"Dataset file '{self.name}' not exists. Please put the file in {self.raw_paths[0]}"
            )

    @property
    def download_dir(self):
        return self.root

    def _process(self) -> dict:
        graph = Graph.from_npz(self.raw_paths[0])
        return dict(graph=graph)

    @property
    def processed_path(self) -> str:
        return None

    @property
    def url(self) -> str:
        return f'{self._url}/{self.name}.npz'

    @property
    def raw_paths(self) -> List[str]:
        return [f"{osp.join(self.download_dir, self.name)}.npz"]

    def list_files(self):
        return self.raw_paths
