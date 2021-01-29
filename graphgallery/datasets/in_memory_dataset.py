import os
import json
import os.path as osp
import numpy as np
import networkx as nx
import pickle as pkl

from typing import Optional, List, Tuple, Callable, Union

from .dataset import Dataset
from ..data.io import makedirs, files_exist, download_file, extract_zip, clean


class InMemoryDataset(Dataset):
    r"""Dataset base class for creating graph datasets which fit completely
    into CPU memory.
    motivated by pytorch_geometric <https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/in_memory_dataset.py>
    """

    def __init__(self,
                 name: Optional[str] = None,
                 root: Optional[str] = None,
                 url: Optional[str] = None,
                 transform=None,
                 verbose: bool = True):
        super().__init__(name, root, url, transform, verbose)

        self.download()
        self.process()

    def download(self) -> None:

        if files_exist(self.raw_paths) or files_exist(self.processed_path):
            if self.verbose:
                print(f"Dataset {self.name} has already existed, loading it.")
                self.show()
            return
        elif files_exist(self.download_paths):
            extract_zip(self.download_paths)
            if self.verbose:
                print(
                    f"Dataset {self.name} has already existed, extracting it."
                )
                self.show()
            return

        if self.verbose:
            print("Downloading...")

        self._download()

        if self.verbose:
            self.show()
            print("Downloading completed.")

    def _download(self):
        makedirs(self.download_dir)

        download_file(self.download_paths, self.urls)
        extract_zip(self.download_paths)
        clean(self.download_paths)

    def process(self) -> None:

        if files_exist(self.processed_path):
            if self.verbose:
                print(f"Processed dataset {self.name} has already existed.")
                self.show(self.processed_path)
            with open(self.processed_path, 'rb') as f:
                cache = pkl.load(f)
        else:
            if self.verbose:
                print("Processing...")
            cache = self._process()
            if self.verbose:
                print("Processing completed.")

        self.graph = self.transform(cache.pop('graph'))
        self.split_cache = cache

    def _process(self) -> dict:
        raise NotImplementedError

    @property
    def url(self) -> str:
        return self._url

    @property
    def download_dir(self) -> str:
        return osp.join(self.root, self.name)

    @property
    def download_paths(self) -> List[str]:
        return self.raw_paths

    @property
    def process_dir(self) -> str:
        return osp.join(self.root, self.name)

    @property
    def processed_filename(self) -> str:
        raise NotImplementedError

    @property
    def processed_path(self) -> str:
        return osp.join(self.process_dir, self.processed_filename)

    @property
    def raw_paths(self) -> List[str]:
        raise NotImplementedError

    @property
    def raw_filenames(self) -> List[str]:
        raise NotImplementedError
