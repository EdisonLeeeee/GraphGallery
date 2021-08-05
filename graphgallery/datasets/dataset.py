import glob
import math
import numpy as np
import os.path as osp

from typing import Union, Optional, List, Tuple, Callable
from tabulate import tabulate

from graphgallery import functional as gf
from ..data.preprocess import (train_val_test_split_tabular,
                               get_train_val_test_split,
                               get_train_val_test_split_gcn)


class Dataset:
    def __init__(self,
                 name,
                 root=None,
                 *,
                 transform=None,
                 verbose=True,
                 url=None):

        if root is None:
            root = 'datasets'

        assert isinstance(root, str), root
        root = osp.abspath(osp.expanduser(root))

        if url:
            self.__url__ = url

        self.root = root
        self.name = str(name)
        self.verbose = verbose

        self._graph = None
        self.split_cache = None
        self.splits = gf.BunchDict()
        self.transform = gf.get(transform)

    @property
    def g(self):
        """alias of graph"""
        return self.graph

    @property
    def graph(self):
        """alias of graph"""
        return self.transform(self._graph)

    @staticmethod
    def available_datasets():
        return dict()

    @property
    def url(self) -> str:
        return self.__url__

    def split_nodes(self,
                    train_size: float = 0.1,
                    val_size: float = 0.1,
                    test_size: float = 0.8,
                    random_state: Optional[int] = None) -> dict:

        assert all((train_size, val_size))
        graph = self.graph
        assert not graph.is_multiple(), "NOT Supported for multiple graph"
        if test_size is None:
            test_size = 1.0 - train_size - val_size
        assert train_size + val_size + test_size <= 1.0

        label = graph.node_label
        train_nodes, val_nodes, test_nodes = train_val_test_split_tabular(
            label.shape[0],
            train_size,
            val_size,
            test_size,
            stratify=label,
            random_state=random_state)
        self.splits.update(
            dict(train_nodes=train_nodes,
                 val_nodes=val_nodes,
                 test_nodes=test_nodes))
        return self.splits

    def split_nodes_as_gcn(self,
                           num_samples: int = 20,
                           random_state: Optional[int] = None) -> dict:
        graph = self.graph
        assert not graph.is_multiple(), "NOT Supported for multiple graph"

        label = graph.node_label
        train_nodes, val_nodes, test_nodes = get_train_val_test_split_gcn(
            label,
            num_samples,
            random_state=random_state)
        self.splits.update(
            dict(train_nodes=train_nodes,
                 val_nodes=val_nodes,
                 test_nodes=test_nodes))
        return self.splits

    def split_nodes_by_sample(self,
                              train_samples_per_class: int,
                              val_samples_per_class: int,
                              test_samples_per_class: int,
                              random_state: Optional[int] = None) -> dict:

        graph = self.graph
        assert not graph.is_multiple(), "NOT Supported for multiple graph"
        self._graph = graph.eliminate_classes(train_samples_per_class + val_samples_per_class).standardize()

        label = self._graph.node_label
        train_nodes, val_nodes, test_nodes = get_train_val_test_split(
            label,
            train_samples_per_class,
            val_samples_per_class,
            test_samples_per_class,
            random_state=random_state)
        self.splits.update(
            dict(train_nodes=train_nodes,
                 val_nodes=val_nodes,
                 test_nodes=test_nodes))
        return self.splits

    def split_edges(self, val_size: float = 0.05,
                    test_size: float = 0.1, train_size: Optional[float] = None,
                    random_state: Optional[int] = None) -> dict:

        graph = self.graph

        assert not graph.is_multiple(), "NOT Supported for multiple graph"
        if train_size is not None:
            train_size = 1 - (val_size + test_size)
            assert train_size + val_size + test_size <= 1
        else:
            assert val_size + test_size < 1

        np.random.seed(random_state)

        is_directed = graph.is_directed()
        graph = graph.to_directed()

        num_nodes = graph.num_nodes
        row, col = graph.edge_index

        splits = gf.BunchDict()

        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

        # TODO: `edge_attr` processing
        edge_attr = getattr(graph, "edge_attr", None)
        if edge_attr is not None:
            edge_attr = edge_attr[mask]

        n_val = int(math.floor(val_size * row.shape[0]))
        n_test = int(math.floor(test_size * row.shape[0]))

        # Positive edges.
        perm = np.random.permutation(row.shape[0])
        row, col = row[perm], col[perm]

        r, c = row[n_val + n_test:], col[n_val + n_test:]
        if train_size is not None:
            n_train = int(math.floor(train_size * row.shape[0]))
            r, c = row[:n_train], col[:n_train]

        splits.train_pos_edge_index = np.stack([r, c], axis=0)

        if not is_directed:
            splits.train_pos_edge_index = gf.asedge(splits.train_pos_edge_index, shape='col_wise', symmetric=True)

        r, c = row[:n_val], col[:n_val]
        splits.val_pos_edge_index = np.stack([r, c], axis=0)

        r, c = row[n_val:n_val + n_test], col[n_val:n_val + n_test]
        splits.test_pos_edge_index = np.stack([r, c], axis=0)

        # Negative edges.
        neg_adj_mask = np.ones((num_nodes, num_nodes), dtype=np.bool)
        neg_adj_mask = np.triu(neg_adj_mask, k=1)
        neg_adj_mask[row, col] = False

        neg_row, neg_col = neg_adj_mask.nonzero()
        perm = np.random.permutation(neg_row.shape[0])[:n_val + n_test]
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        row, col = neg_row[:n_val], neg_col[:n_val]
        splits.val_neg_edge_index = np.stack([row, col], axis=0)

        row, col = neg_row[n_val:n_val + n_test], neg_col[n_val:n_val + n_test]
        splits.test_neg_edge_index = np.stack([row, col], axis=0)

        self.splits.update(**splits)
        return self.splits

    def split_graphs(self,
                     train_size=None,
                     val_size=None,
                     test_size=None,
                     split_by=None,
                     random_state: Optional[int] = None) -> dict:
        raise NotImplementedError

    def show(self, *filepaths) -> None:
        if not filepaths:
            filepaths = self.list_files()

        table_headers = ["File Path", "File Name"]
        items = [osp.split(path) for path in filepaths]
        table = tabulate(items, headers=table_headers,
                         tablefmt="fancy_grid")
        print(f"Files in dataset '{self}':\n" + table)

    def list_files(self):
        return glob.glob(osp.join(self.download_dir, '*'))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, root={self.root})"

    __str__ = __repr__
