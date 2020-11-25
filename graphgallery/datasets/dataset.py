try:
    import texttable
except ImportError:
    texttable = None
import os
import numpy as np
import os.path as osp

from typing import Union, Optional, List, Tuple, Callable

import graphgallery.functional as F
from ..data.preprocess import train_val_test_split_tabular, get_train_val_test_split

Transform = Union[List, Tuple, str, List, Tuple, Callable]


class Dataset:
    def __init__(self, name: str,
                 root: Optional[str] = None,
                 transform: Optional[Transform] = None,
                 verbose: bool = True):
        if root is None:
            root = 'dataset'

        if isinstance(root, str):
            root = osp.expanduser(osp.realpath(root))

        root = osp.abspath(root)
        self.root = root
        self.name = str(name)
        self.verbose = verbose
        self.download_dir = None
        self.process_dir = None
        self.graph = None
        self.split_cache = None
        self.splits = F.Bunch()
        self.transform = F.get(transform)

    @property
    def g(self):
        """alias of graph"""
        return self.graph

    @property
    def urls(self) -> List[str]:
        return [self.url]

    @property
    def url(self) -> str:
        raise NotImplementedError

    @property
    def download_paths(self) -> List[str]:
        return self.raw_paths

    @property
    def raw_paths(self) -> List[str]:
        raise NotImplementedError

    @property
    def processed_path(self) -> List[str]:
        raise NotImplementedError

    @property
    def processed_filename(self):
        raise NotImplementedError

    def download(self) -> None:
        raise NotImplementedError

    def process(self) -> None:
        raise NotImplementedError

    def split_nodes(self, train_size: float = 0.1,
                    val_size: float = 0.1,
                    test_size: float = 0.8,
                    random_state: Optional[int] = None) -> dict:

        assert all((train_size, val_size))
        assert not self.graph.multiple, "NOT Supported for multiple graph"
        if test_size is None:
            test_size = 1.0 - train_size - val_size
        assert train_size + val_size + test_size <= 1.0

        label = self.graph.node_label
        train_nodes, val_nodes, test_nodes = train_val_test_split_tabular(label.shape[0],
                                                                          train_size,
                                                                          val_size,
                                                                          test_size,
                                                                          stratify=label,
                                                                          random_state=random_state)
        self.splits.update(dict(train_nodes=train_nodes,
                                val_nodes=val_nodes,
                                test_nodes=test_nodes))
        return self.splits

    def split_nodes_by_sample(self, train_examples_per_class: int,
                              val_examples_per_class: int,
                              test_examples_per_class: int,
                              random_state: Optional[int] = None) -> dict:

        assert not self.graph.multiple, "NOT Supported for multiple graph"
        self.graph = self.graph.eliminate_classes(train_examples_per_class + val_examples_per_class).standardize()

        label = self.graph.node_label
        train_nodes, val_nodes, test_nodes = get_train_val_test_split(label,
                                                                      train_examples_per_class,
                                                                      val_examples_per_class,
                                                                      test_examples_per_class,
                                                                      random_state=random_state)
        self.splits.update(dict(train_nodes=train_nodes,
                                val_nodes=val_nodes,
                                test_nodes=test_nodes))
        return self.splits

    def split_edges(self, train_size=None,
                    val_size=None,
                    test_size=None,
                    random_state: Optional[int] = None) -> dict:
        raise NotImplementedError

    def split_graphs(self, train_size=None,
                     val_size=None,
                     test_size=None,
                     random_state: Optional[int] = None) -> dict:
        raise NotImplementedError

    @staticmethod
    def show(*filepaths: str) -> None:
        if not filepaths:
            filepaths = self.raw_paths

        if not texttable:
            print(filepaths)
        else:
            t = texttable.Texttable()
            items = [osp.split(path) for path in filepaths]
            t.add_rows([['File Path', 'File Name'], *items])
            print(t.draw())
