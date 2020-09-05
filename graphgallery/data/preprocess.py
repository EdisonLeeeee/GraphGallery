import numpy as np
import scipy.sparse as sp
import networkx as nx
import scipy.sparse as sp
import pickle as pkl

from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, normalize
from sklearn.model_selection import train_test_split


def train_val_test_split_tabular(N,
                                 train_size=0.1,
                                 val_size=0.1,
                                 test_size=0.8,
                                 stratify=None,
                                 random_state=None):

    idx = np.arange(N)
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
        idx_train, idx_val = train_test_split(idx_train_and_val,
                                              random_state=random_state,
                                              train_size=(train_size / (train_size + val_size)),
                                              test_size=(val_size / (train_size + val_size)),
                                              stratify=stratify)

    return idx_train, idx_val, idx_test


def largest_connected_components(sparse_graph, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : Graph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.
    Returns
    -------
    sparse_graph : Graph
        Subgraph of the input graph where only the nodes in largest n_components are kept.
    """
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)


def create_subgraph(sparse_graph, *, nodes_to_remove=None, nodes_to_keep=None):
    """Create a graph with the specified subset of nodes.
    Exactly one of (nodes_to_remove, nodes_to_keep) should be provided, while the other stays None.
    Note that to avoid confusion, it is required to pass node indices as named Parameters to this function.

    Parameters
    ----------
    sparse_graph : Graph
        Input graph.
    nodes_to_remove : array-like of int
        Indices of nodes that have to removed.
    nodes_to_keep : array-like of int
        Indices of nodes that have to be kept.
    Returns
    -------
    sparse_graph : Graph
        Graph with specified nodes removed.
    """
    # Check that Parameters are passed correctly
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError("Only one of nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None:
        nodes_to_keep = [i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    sparse_graph._adj_matrix = sparse_graph._adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph._attr_matrix is not None:
        sparse_graph._attr_matrix = sparse_graph._attr_matrix[nodes_to_keep]
    if sparse_graph._labels is not None:
        sparse_graph._labels = sparse_graph._labels[nodes_to_keep]
    if sparse_graph.node_names is not None:
        sparse_graph.node_names = sparse_graph.node_names[nodes_to_keep]
    return sparse_graph


def binarize_labels(labels, sparse_output=False, return_classes=False):
    """Convert labels vector to a binary label matrix.
    In the default single-label case, labels look like
    labels = [y1, y2, y3, ...].
    Also supports the multi-label format.
    In this case, labels should look something like
    labels = [[y11, y12], [y21, y22, y23], [y31], ...].
    Parameters
    ----------
    labels : array-like, shape [num_samples]
        Array of node labels in categorical single- or multi-label format.
    sparse_output : bool, default False
        Whether return the label_matrix in CSR format.
    return_classes : bool, default False
        Whether return the classes corresponding to the columns of the label matrix.
    Returns
    -------
    label_matrix : np.ndarray or sp.csr_matrix, shape [num_samples, num_classes]
        Binary matrix of class labels.
        num_classes = number of unique values in "labels" array.
        label_matrix[i, k] = 1 <=> node i belongs to class k.
    classes : np.array, shape [num_classes], optional
        Classes that correspond to each column of the label_matrix.
    """
    if hasattr(labels[0], '__iter__'):  # labels[0] is iterable <=> multilabel format
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
    else:
        binarizer = LabelBinarizer(sparse_output=sparse_output)
    label_matrix = binarizer.fit_transform(labels).astype(np.float32)
    return (label_matrix, binarizer.classes_) if return_classes else label_matrix


def remove_underrepresented_classes(g, train_examples_per_class, val_examples_per_class):
    """Remove nodes from graph that correspond to a class of which there are less than
    num_classes * train_examples_per_class + num_classes * val_examples_per_class nodes.
    Those classes would otherwise break the training procedure.
    """
    min_examples_per_class = train_examples_per_class + val_examples_per_class
    examples_counter = Counter(g.labels)
    keep_classes = set(class_ for class_, count in examples_counter.items() if count > min_examples_per_class)
    keep_indices = [i for i in range(len(g.labels)) if g.labels[i] in keep_classes]

    return create_subgraph(g, nodes_to_keep=keep_indices)


def get_train_val_test_split(stratify,
                             train_examples_per_class,
                             val_examples_per_class,
                             test_examples_per_class=None,
                             random_state=None):

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


def sample_per_class(stratify, num_examples_per_class,
                     forbidden_indices=None, random_state=None):

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
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])
