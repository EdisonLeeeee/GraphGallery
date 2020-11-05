import numpy as np
import networkx as nx
import scipy.sparse as sp
import pickle as pkl

from typing import Optional, List, Tuple
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, normalize
from sklearn.model_selection import train_test_split
from graphgallery.typing import MultiArrayLike, ArrayLike1D, GalleryGraph


def train_val_test_split_tabular(N: int,
                                 train_size: float = 0.1,
                                 val_size: float = 0.1,
                                 test_size: float = 0.8,
                                 stratify: Optional[ArrayLike1D] = None,
                                 random_state: Optional[int] = None) -> MultiArrayLike:
    """
    Train a train_tab_test_tab_test.

    Args:
        N: (todo): write your description
        train_size: (int): write your description
        val_size: (int): write your description
        test_size: (int): write your description
        stratify: (str): write your description
        random_state: (int): write your description
    """

    idx = np.arange(N)
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(
                                                       train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)

    stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(
                                              train_size / (train_size + val_size)),
                                          test_size=(
                                              val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test


def largest_connected_components(graph: GalleryGraph, n_components: int = 1) -> GalleryGraph:
    """Select the largest connected components in the graph.

    Parameters
    ----------
    graph : GalleryGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.
    Returns
    -------
    graph : GalleryGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.
    """
    _, component_indices = sp.csgraph.connected_components(
        graph.adj_matrix)
    component_sizes = np.bincount(component_indices)
    # reverse order to sort descending
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return create_subgraph(graph, nodes_to_keep=nodes_to_keep)


def create_subgraph(graph: GalleryGraph, *,
                    nodes_to_remove: Optional[ArrayLike1D] = None,
                    nodes_to_keep: Optional[ArrayLike1D] = None) -> GalleryGraph:
    """Create a graph with the specified subset of nodes.
    Exactly one of (nodes_to_remove, nodes_to_keep) should be provided, while the other stays None.
    Note that to avoid confusion, it is required to pass node indices as named Parameters to this function.

    Parameters
    ----------
    graph : GalleryGraph
        Input graph.
    nodes_to_remove : array-like of int
        Indices of nodes that have to removed.
    nodes_to_keep : array-like of int
        Indices of nodes that have to be kept.
    Returns
    -------
    graph : GalleryGraph
        GalleryGraph with specified nodes removed.
    """
    # Check that Parameters are passed correctly
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError(
            "Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError(
            "Only one of nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None:
        if len(nodes_to_remove) == 0:
            return graph.copy()
        nodes_to_keep = np.setdiff1d(np.arange(graph.n_nodes), nodes_to_remove)
    elif nodes_to_keep is not None:
        nodes_to_keep = np.sort(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    graph = graph.copy()
    adj_matrix, attr_matrix, labels = graph.raw()
    graph.adj_matrix = adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if attr_matrix is not None:
        graph.attr_matrix = attr_matrix[nodes_to_keep]
    if labels is not None:
        graph.labels = labels[nodes_to_keep]
    if graph.node_names is not None:
        graph.node_names = graph.node_names[nodes_to_keep]
    return graph


def binarize_labels(labels: ArrayLike1D, sparse_output: bool = False, return_classes: bool = False):
    """Convert labels vector to a binary label matrix.
    In the default single-label case, labels look like
    labels = [y1, y2, y3, ...].
    Also supports the multi-label format.
    In this case, labels should look something like
    labels = [[y11, y12], [y21, y22, y23], [y31], ...].
    Parameters
    ----------
    labels : array-like, shape [n_samples]
        Array of node labels in categorical single- or multi-label format.
    sparse_output : bool, default False
        Whether return the label_matrix in CSR format.
    return_classes : bool, default False
        Whether return the classes corresponding to the columns of the label matrix.
    Returns
    -------
    label_matrix : np.ndarray or sp.csr_matrix, shape [n_samples, n_classes]
        Binary matrix of class labels.
        n_classes = number of unique values in "labels" array.
        label_matrix[i, k] = 1 <=> node i belongs to class k.
    classes : np.array, shape [n_classes], optional
        Classes that correspond to each column of the label_matrix.
    """
    if hasattr(labels[0], '__iter__'):  # labels[0] is iterable <=> multilabel format
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
    else:
        binarizer = LabelBinarizer(sparse_output=sparse_output)
    label_matrix = binarizer.fit_transform(labels).astype(np.float32)
    return (label_matrix, binarizer.classes_) if return_classes else label_matrix


def get_train_val_test_split(stratify: ArrayLike1D,
                             train_examples_per_class: int,
                             val_examples_per_class: int,
                             test_examples_per_class: Optional[None] = None,
                             random_state: Optional[None] = None) -> MultiArrayLike:
    """
    Return a set.

    Args:
        stratify: (todo): write your description
        train_examples_per_class: (todo): write your description
        val_examples_per_class: (todo): write your description
        test_examples_per_class: (todo): write your description
        Optional: (todo): write your description
        random_state: (int): write your description
        Optional: (todo): write your description
    """

    random_state = np.random.RandomState(random_state)
    remaining_indices = list(range(stratify.shape[0]))

    idx_train = sample_per_class(stratify, train_examples_per_class,
                                 random_state=random_state)

    idx_val = sample_per_class(stratify, val_examples_per_class,
                               random_state=random_state,
                               forbidden_indices=idx_train)
    forbidden_indices = np.concatenate((idx_train, idx_val))

    if test_examples_per_class is not None:
        idx_test = sample_per_class(stratify, test_examples_per_class,
                                    random_state=random_state,
                                    forbidden_indices=forbidden_indices)
    else:
        idx_test = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(idx_train)) == len(idx_train)
    assert len(set(idx_val)) == len(idx_val)
    assert len(set(idx_test)) == len(idx_test)
    # assert sets are mutually exclusive
    assert len(set(idx_train) - set(idx_val)) == len(set(idx_train))
    assert len(set(idx_train) - set(idx_test)) == len(set(idx_train))
    assert len(set(idx_val) - set(idx_test)) == len(set(idx_val))

    return idx_train, idx_val, idx_test


def sample_per_class(stratify: ArrayLike1D, n_examples_per_class: int,
                     forbidden_indices: Optional[ArrayLike1D] = None,
                     random_state: Optional[int] = None) -> ArrayLike1D:
    """
    Generate samples from the dataset.

    Args:
        stratify: (todo): write your description
        n_examples_per_class: (int): write your description
        forbidden_indices: (todo): write your description
        random_state: (int): write your description
    """

    n_classes = stratify.max() + 1
    n_samples = stratify.shape[0]
    sample_indices_per_class = {index: [] for index in range(n_classes)}

    # get indices sorted by class
    for class_index in range(n_classes):
        for sample_index in range(n_samples):
            if stratify[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], n_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def parse_index_file(filename: str) -> List:
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def process_planetoid_datasets(name: str, paths: List[str]) -> Tuple:
    """
    Process a list of matrices.

    Args:
        name: (str): write your description
        paths: (str): write your description
    """
    objs = []
    for fname in paths:
        with open(fname, 'rb') as f:
            try:
                obj = pkl.load(f, encoding='latin1')
            except pkl.PickleError:
                obj = parse_index_file(fname)

            objs.append(obj)

    x, tx, allx, y, ty, ally, graph, test_idx_reorder = objs
    test_idx_range = np.sort(test_idx_reorder)

    if name.lower() == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = np.arange(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    attr_matrix = sp.vstack((allx, tx)).tolil()
    attr_matrix[test_idx_reorder, :] = attr_matrix[test_idx_range, :]

    adj_matrix = nx.adjacency_matrix(nx.from_dict_of_lists(
        graph, create_using=nx.DiGraph()))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_train = np.arange(len(y))
    idx_val = np.arange(len(y), len(y) + 500)
    idx_test = test_idx_range

    labels = labels.argmax(1)

    adj_matrix = adj_matrix.astype('float32')
    attr_matrix = attr_matrix.astype('float32')

    return adj_matrix, attr_matrix, labels, idx_train, idx_val, idx_test
