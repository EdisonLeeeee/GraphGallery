import os
import os.path as osp
import numpy as np
import pickle as pkl


from graphgallery.data import Dataset
from graphgallery.data.io import makedirs, files_exist, download_file
from graphgallery.data.preprocess import process_planetoid_datasets
from graphgallery.data.graph import Graph


DATASETS = {'citeseer', 'cora', 'pubmed'}


class Planetoid(Dataset):
    """The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    The original url is: <https://github.com/kimiyoung/planetoid/raw/master/data>
    """

    github_url = "https://raw.githubusercontent.com/EdisonLeeeee/GraphData/master/datasets/planetoid"
    supported_datasets = DATASETS

    def __init__(self, name, root=None, verbose=True):
        name = name.lower()

        if not name in self.supported_datasets:
            raise ValueError(f"Currently only support for these datasets {self.supported_datasets}.")

        super().__init__(name, root, verbose)

        self.download_dir = osp.join(self.root, 'planetoid')

        makedirs(self.download_dir)

        self.download()
        self.process()

    def download(self):

        if files_exist(self.raw_paths):
            print(f"Downloaded dataset files have existed.")
            if self.verbose:
                self.print_files(self.raw_paths)
            return

        print("Downloading...")
        download_file(self.raw_paths, self.urls)
        if self.verbose:
            self.print_files(self.raw_paths)
        print("Downloading completed.")

    def process(self):

        print("Processing...")
        adj, attributes, labels, idx_train, idx_val, idx_test = process_planetoid_datasets(self.name, self.raw_paths)
        self.graph = Graph(adj, attributes, labels).eliminate_self_loops()
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        print("Processing completed.")

    def split(self, train_size=None, val_size=None, test_size=None,
              random_state=None):
        if train_size is None:
            return self.idx_train, self.idx_val, self.idx_test
        else:
            return super().split(train_size, val_size, test_size, random_state)

    @property
    def urls(self):
        return [f"{osp.join(self.github_url, raw_file_name)}" for raw_file_name in self.raw_file_names]

    @property
    def raw_file_names(self):
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return ['ind.{}.{}'.format(self.name.lower(), name) for name in names]

    @property
    def raw_paths(self):
        return [f"{osp.join(self.download_dir, raw_file_name)}" for raw_file_name in self.raw_file_names]
