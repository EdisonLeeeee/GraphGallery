import glob
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
            self._url = url

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
        return self._url

    def split_nodes(self,
                    train_size: float = 0.1,
                    val_size: float = 0.1,
                    test_size: float = 0.8,
                    random_state: Optional[int] = None) -> dict:

        assert all((train_size, val_size))
        graph = self.graph
        assert not graph.multiple, "NOT Supported for multiple graph"
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
        assert not graph.multiple, "NOT Supported for multiple graph"

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
        assert not graph.multiple, "NOT Supported for multiple graph"
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

    def split_edges(self,
                    train_size=None,
                    val_size=None,
                    test_size=None,
                    random_state: Optional[int] = None) -> dict:
        raise NotImplementedError

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
