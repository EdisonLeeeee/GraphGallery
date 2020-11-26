import os
import json
import os.path as osp
import numpy as np
import networkx as nx
import pickle as pkl

from typing import Optional, List, Tuple, Callable, Union

from .dataset import Dataset
from ..data.io import makedirs, files_exist, download_file, extract_zip, clean, load_npz

Transform = Union[List, Tuple, str, List, Tuple, Callable]


class InMemoryDataset(Dataset):
    r"""Dataset base class for creating graph datasets which fit completely
    into CPU memory.
    motivated by pytorch_geometric <https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/in_memory_dataset.py>
    """

    def __init__(self, name: Optional[str] = None,
                 root: Optional[str] = None,
                 transform: Optional[Transform] = None,
                 verbose: bool = True):
        super().__init__(name, root, transform, verbose)

        makedirs(self.download_dir)
        # since they are same.
        # makedirs(self.process_dir)
        self.download()
        self.process()

    def download(self) -> None:

        if files_exist(self.raw_paths) or files_exist(self.processed_path):
            if self.verbose:
                print(f"Dataset {self.name} have already existed, loading it.")
                self.show(*self.raw_paths)
            return
        elif files_exist(self.download_paths):
            extract_zip(self.download_paths)
            if self.verbose:
                print(f"Dataset {self.name} have already existed, extracting it.")
                self.show(*self.raw_paths)
            return

        if self.verbose:
            print("Downloading...")
        try:
            download_file(self.download_paths, self.urls)
        except Exception:
            raise Exception(f"404 not Found, maybe you specified a wrong dataset '{self.name}'? (NOTE, it is 'Case Sensitive' for dataset name)")

        extract_zip(self.download_paths, self.extract_folder)
        clean(self.download_paths)

        if self.verbose:
            self.show(*self.raw_paths)
            print("Downloading completed.")

    @property
    def extract_folder(self):
        return None

    def process(self) -> None:

        if files_exist(self.processed_path):
            if self.verbose:
                print(f"Processed dataset {self.name} have already existed.")
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

    def _process(self) -> None:
        raise NotImplementedError

    @property
    def url(self) -> str:
        return self._url

    @property
    def processed_filename(self) -> str:
        raise NotImplementedError

    @property
    def processed_path(self) -> str:
        return osp.join(self.process_dir, self.processed_filename)

    @property
    def raw_filenames(self) -> List[str]:
        raise NotImplementedError

    @property
    def download_paths(self):
        raise NotImplementedError

    @property
    def raw_paths(self) -> List[str]:
        raise NotImplementedError
