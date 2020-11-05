try:
    import texttable
except ImportError:
    texttable = None
import numpy as np
import os.path as osp

from abc import ABC

from typing import Union, Optional, List, Tuple
from graphgallery.data.preprocess import train_val_test_split_tabular, get_train_val_test_split

class Dataset(ABC):
    def __init__(self, name: str, root: Optional[str]=None, verbose: bool=True):
        """
        Initialize the graph.

        Args:
            self: (todo): write your description
            name: (str): write your description
            root: (str): write your description
            verbose: (bool): write your description
        """
        if root is None:
            root = 'dataset'

        if isinstance(root, str):
            root = osp.expanduser(osp.normpath(root))

        root = osp.abspath(root)
        self.root = root
        self.name = str(name).lower()
        self.verbose = verbose
        self.download_dir = None
        self.processed_dir = None
        self.graph = None

    @property
    def urls(self) -> List[str]:
        """
        A list of urls.

        Args:
            self: (todo): write your description
        """
        return [self.url]

    @property
    def url(self) -> str:
        """
        Returns the url.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError

    def download(self) -> None:
        """
        Downloads the file.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError

    def process(self) -> None:
        """
        Process the given process.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError

    def split(self, train_size: float = 0.1, val_size: float = 0.1, test_size: float = 0.8,
              random_state: Optional[int]=None) -> Tuple[np.ndarray]:
        """
        Return a list of the dataset.

        Args:
            self: (todo): write your description
            train_size: (int): write your description
            val_size: (int): write your description
            test_size: (int): write your description
            random_state: (int): write your description
        """

        assert all((train_size, val_size))
        if test_size is None:
            test_size = 1.0 - train_size - val_size
        assert train_size+val_size+test_size<=1.0

        labels = self.graph.labels
        idx_train, idx_val, idx_test = train_val_test_split_tabular(labels.shape[0],
                                                                    train_size,
                                                                    val_size,
                                                                    test_size,
                                                                    stratify=labels,
                                                                    random_state=random_state)

        return idx_train, idx_val, idx_test

    def split_by_sample(self, train_examples_per_class: int,
                        val_examples_per_class: int,
                        test_examples_per_class: int,
                        random_state: Optional[int]=None)  -> Tuple[np.ndarray]:
        """
        Split the dataset according to the given training set.

        Args:
            self: (todo): write your description
            train_examples_per_class: (int): write your description
            val_examples_per_class: (float): write your description
            test_examples_per_class: (todo): write your description
            random_state: (int): write your description
        """

        self.graph = self.graph.eliminate_classes(train_examples_per_class+val_examples_per_class).standardize()
            
        labels = self.graph.labels
        idx_train, idx_val, idx_test = get_train_val_test_split(labels,
                                                                train_examples_per_class,
                                                                val_examples_per_class,
                                                                test_examples_per_class,
                                                                random_state=random_state)
        return idx_train, idx_val, idx_test

    @staticmethod
    def print_files(filepaths: List[str]) -> None:
        """
        Prints a table.

        Args:
            filepaths: (str): write your description
        """
        if not texttable:
            print(filepaths)
        else:
            t = texttable.Texttable()
            items = [(path.split('/')[-1], path) for path in filepaths]

            t.add_rows([['File Name', 'File Path'], *items])
            print(t.draw())
