import os
import glob
import numpy as np
import os.path as osp

from typing import Union, Optional, List, Tuple, Callable
from tabulate import tabulate

from graphgallery import functional as gf
from ..data.preprocess import train_val_test_split_tabular, get_train_val_test_split

Transform = Union[List, Tuple, str, List, Tuple, Callable]


class Dataset:
    def __init__(self,
                 name: str,
                 root: Optional[str] = None,
                 url: Optional[str] = None,
                 transform: Optional[Transform] = None,
                 verbose: bool = True):

        if root is None:
            root = 'dataset'

        assert isinstance(root, str), root
        root = osp.abspath(osp.expanduser(root))

        if url:
            self._url = url

        self.root = root
        self.name = str(name)
        self.verbose = verbose

        self.graph = None
        self.split_cache = None
        self.splits = gf.BunchDict()
        self.transform = gf.get(transform)

    @property
    def g(self):
        """alias of graph"""
        return self.graph

    @staticmethod
    def available_datasets():
        return None

    @property
    def urls(self) -> List[str]:
        return [self.url]

    @property
    def url(self) -> str:
        return self._url

    def split_nodes(self,
                    train_size: float = 0.1,
                    val_size: float = 0.1,
                    test_size: float = 0.8,
                    random_state: Optional[int] = None) -> dict:

        assert all((train_size, val_size))
        assert not self.graph.multiple, "NOT Supported for multiple graph"
        if test_size is None:
            test_size = 1.0 - train_size - val_size
        assert train_size + val_size + test_size <= 1.0

        label = self.graph.node_label
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

    def split_nodes_by_sample(self,
                              trainum_examples_per_class: int,
                              val_examples_per_class: int,
                              test_examples_per_class: int,
                              random_state: Optional[int] = None) -> dict:

        assert not self.graph.multiple, "NOT Supported for multiple graph"
        self.graph = self.graph.eliminate_classes(
            trainum_examples_per_class + val_examples_per_class).standardize()

        label = self.graph.node_label
        train_nodes, val_nodes, test_nodes = get_train_val_test_split(
            label,
            trainum_examples_per_class,
            val_examples_per_class,
            test_examples_per_class,
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

    def show(self, *filepaths: str) -> None:
        if not filepaths:
            filepaths = self.list_files()

        table_headers = ["File Path", "File Name"]
        items = [osp.split(path) for path in filepaths]
        table = tabulate(items, headers=table_headers,
                         tablefmt="fancy_grid"
                         )
        print(f"Files in dataset {self.__class__.__name__}:\n" + table)

    def list_files(self):
        return glob.glob(osp.join(self.download_dir, '*'))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.graph})"
    __str__ = __repr__
