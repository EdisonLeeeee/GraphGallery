from abc import ABC
from copy import copy as _copy, deepcopy as _deepcopy
from typing import Union, Tuple
from graphgallery.typing import SparseMatrix, MultiSparseMatrix, ArrayLike, MultiArrayLike, GalleryGraph


class BaseGraph(ABC):

    def __init__(self):
        """
        Initialize the class

        Args:
            self: (todo): write your description
        """
        ...

    @property
    def n_nodes(self) -> int:
        """
        Returns a list of nodes in the node belongs to.

        Args:
            self: (todo): write your description
        """
        ...

    @property
    def n_edges(self) -> int:
        """
        The number of edges in the graph.

        Args:
            self: (todo): write your description
        """
        ...

    @property
    def n_graphs(self) -> int:
        """
        Returns the number of integers.

        Args:
            self: (todo): write your description
        """
        ...

    @property
    def n_attrs(self) -> int:
        """
        Returns the number of attributes.

        Args:
            self: (todo): write your description
        """
        ...

    @property
    def n_classes(self) -> int:
        """
        Return the number of classes.

        Args:
            self: (todo): write your description
        """
        ...

    @property
    def A(self) -> Union[SparseMatrix, MultiSparseMatrix]:
        """alias of adj_matrix."""
        return self.adj_matrix

    @property
    def X(self) -> Union[ArrayLike, MultiArrayLike]:
        """alias of attr_matrix."""
        return self.attr_matrix

    @property
    def Y(self) -> Union[ArrayLike, MultiArrayLike]:
        """alias of labels."""
        return self.labels

    @property
    def D(self) -> Union[Tuple[ArrayLike, ArrayLike], ArrayLike, MultiArrayLike]:
        """alias of degrees."""
        return self.degrees

    def __len__(self) -> int:
        """
        Returns the number of nodes.

        Args:
            self: (todo): write your description
        """
        return self.n_graphs

    def unpack(self) -> tuple:
        """
        Unpack a tuple.

        Args:
            self: (todo): write your description
        """
        return self.A, self.X, self.Y

    def raw(self) -> tuple:
        """Return the raw (A, X, Y) triplet."""
        return self._adj_matrix, self._attr_matrix, self._labels

    def is_labeled(self) -> bool:
        """
        Returns true if the labeled.

        Args:
            self: (todo): write your description
        """
        return self.labels is not None

    def is_attributed(self) -> bool:
        """
        Return true if the node is a : class attribute.

        Args:
            self: (todo): write your description
        """
        return self.attr_matrix is not None

    def copy(self, deepcopy: bool = False) -> GalleryGraph:
        """
        Returns a copy of this instance.

        Args:
            self: (todo): write your description
            deepcopy: (bool): write your description
        """
        cls = self.__class__
        if deepcopy:
            return _deepcopy(self)
        else:
            return _copy(self)

    def __copy__(self) -> GalleryGraph:
        """
        Returns a copy of this instance.

        Args:
            self: (todo): write your description
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> GalleryGraph:
        """
        Return a copy of this instance.

        Args:
            self: (todo): write your description
            memo: (dict): write your description
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, _deepcopy(v, memo))
        return result
