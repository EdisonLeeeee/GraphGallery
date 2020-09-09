import os
import zipfile
import os.path as osp
import numpy as np

from graphgallery.data import Dataset
from graphgallery.data.io import makedirs, files_exist, download_file
from graphgallery.data.graph import Graph, load_dataset


_DATASETS = ('citeseer', 'cora', 'cora_ml', 'cora_full', 'amazon_cs', 'amazon_photo',
             'coauthor_cs', 'coauthor_phy', 'polblogs', 'pubmed', 'reddit')


class NPZDataset(Dataset):

    github_url = "https://raw.githubusercontent.com/EdisonLeeeee/GraphData/master/datasets/npz/{}.npz"
    supported_datasets = _DATASETS

    def __init__(self, name, root=None, url=None, standardize=False, verbose=True):
        name = name.lower()

        if not name in self.supported_datasets:
            raise ValueError(
                f"Currently only support for these datasets {self.supported_datasets}.")

        super().__init__(name, root, verbose)

        self._url = url
        self.download_dir = osp.join(self.root, "npz")
        self.standardize = standardize

        makedirs(self.download_dir)

        self.download()
        self.process()

    def download(self):

        if files_exist(self.raw_paths):
            print(f"Downloaded dataset files have existed.")
            if self.verbose:
                self.print_files(self.raw_paths)
            return

        self.print_files(self.raw_paths)

        print("Downloading...")
        download_file(self.raw_paths, self.urls)
        if self.verbose:
            self.print_files(self.raw_paths)
        print("Downloading completed.")

    def process(self):

        print("Processing...")
        graph = load_dataset(
            self.raw_paths[0]).eliminate_selfloops().to_undirected()
        if self.standardize:
            graph = graph.standardize()
        self.graph = graph
        print("Processing completed.")

    @property
    def url(self):
        if isinstance(self._url, str):
            return self._url
        else:
            return self.github_url.format(self.name)

    @property
    def raw_paths(self):
        return [f"{osp.join(self.download_dir, self.name)}.npz"]
