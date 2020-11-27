import os
import sys
import zipfile
import os.path as osp
import numpy as np

from typing import Optional, List, Tuple, Union, Callable
from .dataset import Dataset
from ..data.io import makedirs, files_exist, download_file
from ..data.graph import Graph


_DATASETS = ('citeseer', 'citeseer_full', 'cora', 'cora_ml',
             'cora_full', 'amazon_cs', 'amazon_photo',
             'coauthor_cs', 'coauthor_phy', 'polblogs', 'karate_club',
             'pubmed', 'flickr', 'blogcatalog', 'dblp', 'acm', 'uai')

_DATASET_URL = "https://raw.githubusercontent.com/EdisonLeeeee/GraphData/master/datasets/"

Transform = Union[List, Tuple, str, List, Tuple, Callable]


class NPZDataset(Dataset):

    _url = _DATASET_URL

    def __init__(self, name: str,
                 root: Optional[str] = None,
                 url: Optional[str] = None,
                 transform: Optional[Transform] = None,
                 verbose: bool = True
                 ):

        name = str(name)
        if not name in self.available_datasets():
            print(f"Dataset not found in available datasets. Using custom dataset: {name}.", file=sys.stderr)
            custom = True
        else:
            custom = False

        super().__init__(name, root, url, transform, verbose)

        if not custom:
            makedirs(self.download_dir)
            self.download()
        elif not osp.exists(self.raw_paths[0]):
            raise RuntimeError(f"Dataset file '{name}' not exists. Please put the file in {self.raw_paths[0]}")

        self.process()

    @staticmethod
    def available_datasets():
        return _DATASETS

    def download(self) -> None:

        if files_exist(self.raw_paths):
            if self.verbose:
                print(f"Dataset {self.name} have already existed.")
                self.show(*self.raw_paths)
            return

        if self.verbose:
            print("Downloading...")
        download_file(self.download_paths, self.urls)
        if self.verbose:
            self.show(*self.raw_paths)
            print("Downloading completed.")

    @property
    def download_dir(self):
        return self.root

    def process(self) -> None:
        if self.verbose:
            print("Processing...")
        graph = Graph.from_npz(self.raw_paths[0])
        # TODO
#             self.raw_paths[0]).eliminate_selfloops().to_unweighted().to_undirected()
        # if self.standardize:
        #     graph = graph.standardize()
        self.graph = self.transform(graph)
        if self.verbose:
            print("Processing completed.")

    @property
    def url(self) -> str:
        return '{}/{}.zip'.format(self._url, self.name)

    @property
    def raw_paths(self) -> List[str]:
        return [f"{osp.join(self.download_dir, self.name)}.npz"]
