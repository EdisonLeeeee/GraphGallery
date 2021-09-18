import os.path as osp
import numpy as np
import scipy.sparse as sp
import pickle as pkl

from typing import Optional, List
from graphgallery import functional as gf

from .in_memory_dataset import InMemoryDataset
from ..data.graph import Graph


class Reddit(InMemoryDataset):
    r"""The Reddit dataset from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, containing
    Reddit posts belonging to different communities..
    """

    __url__ = 'https://data.dgl.ai/dataset/reddit.zip'

    def __init__(self,
                 root=None,
                 *,
                 transform=None,
                 verbose=True,
                 url=None,
                 remove_download=True):

        super().__init__(name="reddit", root=root,
                         transform=transform,
                         verbose=verbose, url=url,
                         remove_download=remove_download)

    @staticmethod
    def available_datasets():
        return gf.BunchDict(reddit="reddit dataset")

    def __process__(self):

        data = np.load(osp.join(self.download_dir, 'reddit_data.npz'))
        adj_matrix = sp.load_npz(
            osp.join(self.download_dir, 'reddit_graph.npz')).tocsr(copy=False)

        node_attr = data['feature']
        node_label = data['label']
        node_graph_label = data['node_types']
        graph = Graph(adj_matrix,
                      node_attr,
                      node_label,
                      node_graph_label=node_graph_label)

        train_nodes = np.where(node_graph_label == 1)[0]
        val_nodes = np.where(node_graph_label == 2)[0]
        test_nodes = np.where(node_graph_label == 3)[0]

        cache = dict(train_nodes=train_nodes,
                     val_nodes=val_nodes,
                     test_nodes=test_nodes,
                     graph=graph)
        with open(self.process_path, 'wb') as f:
            pkl.dump(cache, f)
        return cache

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
    def process_filename(self):
        return f'{self.name}.pkl'

    @property
    def raw_filenames(self) -> List[str]:
        return ['reddit_data.npz', 'reddit_graph.npz']

    @property
    def download_paths(self):
        return [osp.join(self.download_dir, self.name + '.zip')]

    @property
    def raw_paths(self) -> List[str]:
        return [
            osp.join(self.download_dir, raw_filename)
            for raw_filename in self.raw_filenames
        ]
