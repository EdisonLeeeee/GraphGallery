import os
import glob
import requests
import os.path as osp
import numpy as np
import pickle as pkl
import pandas as pd
from urllib.error import URLError
from typing import Optional, List

from .in_memory_dataset import InMemoryDataset
from ..data.edge_graph import EdgeGraph
from ..data.io import makedirs, extractall, remove

_DATASET_URL = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets'
_DATASET_CLEAN_URL = 'https://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets'


class TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    """

    def __init__(self,
                 name,
                 root=None,
                 *,
                 transform=None,
                 verbose=True,
                 url=None,
                 remove_download=True):

        if name.endswith('_clean'):
            name = name[:-6]
            self.__url__ = _DATASET_CLEAN_URL
        else:
            self.__url__ = _DATASET_URL

        super().__init__(name=name, root=root,
                         transform=transform,
                         verbose=verbose, url=url,
                         remove_download=remove_download)

    @staticmethod
    def available_datasets():
        try:
            return [
                d[:-4] for d in pd.read_html(_DATASET_URL)
                [0].Name[2:-1].values.tolist()
            ]
        except URLError:
            # No internet, don't panic
            print('No connection. See {}'.format(_DATASET_URL))
            return []

    def __download__(self):
        req = requests.get(self.url)
        if req.status_code == 404:
            raise ValueError(
                f"Unknown dataset {self.name}. See '{self.__class__.__name__}.available_datasets()'"
                " for a list of available datasets.")
        makedirs(self.download_dir)

        with open(self.download_paths[0], 'wb') as f:
            f.write(req.content)

        extractall(self.download_paths, osp.split(self.download_dir)[0])

        if self.remove_download:
            remove(self.download_paths)

    def __process__(self):
        folder = self.download_dir
        prefix = self.name
        files = glob.glob(osp.join(folder, f'{prefix}_*.txt'))
        names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]
        edge_index = genfromtxt(osp.join(folder, prefix + '_A.txt'),
                                dtype=np.int64).T - 1
        node_graph_label = genfromtxt(osp.join(folder, prefix + '_graph_indicator.txt'),
                                      dtype=np.int64) - 1
        edge_graph_label = node_graph_label[edge_index[0]]
        node_attr = node_label = None
        if 'node_attributes' in names:
            node_attr = genfromtxt(osp.join(folder,
                                            prefix + '_node_attributes.txt'),
                                   dtype=np.float32)

        if 'node_labels' in names:
            node_label = genfromtxt(osp.join(folder,
                                             prefix + '_node_labels.txt'),
                                    dtype=np.int64)
            node_label = node_label - node_label.min(0)

        edge_attr = edge_label = None
        if 'edge_attributes' in names:
            edge_attr = genfromtxt(osp.join(folder,
                                            prefix + '_edge_attributes.txt'),
                                   dtype=np.float32)
        if 'edge_labels' in names:
            edge_label = genfromtxt(osp.join(folder,
                                             prefix + '_edge_labels.txt'),
                                    dtype=np.int64)
            edge_label = edge_label - edge_label.min(0)

        graph_attr = graph_label = None
        if 'graph_attributes' in names:  # Regression problem.
            graph_attr = np.genfromtxt(osp.join(
                folder, prefix + '_graph_attributes.txt'),
                dtype=np.float32)
        if 'graph_labels' in names:  # Classification problem.
            graph_label = np.genfromtxt(osp.join(folder,
                                                 prefix + '_graph_labels.txt'),
                                        dtype=np.int64)
            _, graph_label = np.unique(graph_label, return_inverse=True)

        graph = EdgeGraph(edge_index,
                          edge_attr=edge_attr,
                          edge_label=edge_label,
                          edge_graph_label=edge_graph_label,
                          node_attr=node_attr,
                          node_label=node_label,
                          node_graph_label=node_graph_label,
                          graph_attr=graph_attr,
                          graph_label=graph_label)

        cache = {'graph': graph}
        with open(self.process_path, 'wb') as f:
            pkl.dump(cache, f)
        return cache

    @property
    def download_dir(self):
        return osp.join(self.root, "TU", self.name)

    def split_graphs(self,
                     train_size=None,
                     val_size=None,
                     test_size=None,
                     split_by=None,
                     random_state: Optional[int] = None):
        raise NotImplementedError

    @property
    def url(self) -> str:
        return '{}/{}.zip'.format(self.__url__, self.name)

    @property
    def process_filename(self):
        return f'{self.name}.pkl'

    @property
    def raw_filenames(self) -> List[str]:
        names = ['A', 'graph_indicator']  # and more
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def download_paths(self):
        return [osp.join(self.download_dir, self.name + '.zip')]

    @property
    def raw_paths(self) -> List[str]:
        return [
            osp.join(self.download_dir, raw_filename)
            for raw_filename in self.raw_filenames
        ]


def genfromtxt(path, sep=',', start=0, end=None, dtype=None, device=None):
    # with open(path, 'r') as f:
    #     src = f.read().split('\n')[:-1]

    # src = [[float(x) for x in line.split(sep)[start:end]] for line in src]
    # src = np.asarray(src, dtype=dtype).squeeze()

    # # return src
    return np.loadtxt(path, delimiter=sep).astype(dtype).squeeze()
