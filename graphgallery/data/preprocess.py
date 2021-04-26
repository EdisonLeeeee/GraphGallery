import copy
import numpy as np
import networkx as nx
import scipy.sparse as sp
import pickle as pkl

from typing import Optional, List, Tuple, Union
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, normalize
from sklearn.model_selection import train_test_split


def train_val_test_split_tabular(N: int,
                                 train_size: float = 0.1,
                                 val_size: float = 0.1,
                                 test_size: float = 0.8,
                                 stratify=None,
                                 random_state: Optional[int] = None):

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


def binarize_labels(labels, sparse_output: bool = False, returnum_node_classes: bool = False):
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
    returnum_node_classes : bool, default False
        Whether return the classes corresponding to the columns of the label matrix.

    Returns
    -------
    label_matrix : np.ndarray or sp.csr_matrix, shape [num_samples, num_node_classes]
        Binary matrix of class labels.
        num_node_classes = number of unique values in "labels" array.
        label_matrix[i, k] = 1 <=> node i belongs to class k.
    classes : np.array, shape [num_node_classes], optional
        Classes that correspond to each column of the label_matrix.
    """
    if hasattr(labels[0], '__iter__'):  # labels[0] is iterable <=> multilabel format
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
    else:
        binarizer = LabelBinarizer(sparse_output=sparse_output)
    label_matrix = binarizer.fit_transform(labels).astype(np.float32)
    return (label_matrix, binarizer.classes_) if returnum_node_classes else label_matrix



def get_train_val_test_split_gcn(labels, num_samples=20, random_state=None):
    """This setting follows gcn, where we randomly sample 'num_samples' instances for each class as training data, 500 instances as validation data, 1000 instances as test data.

    Note here we are not using fixed splits. When random_state changes, the splits ofwill also change.

    Parameters
    ----------
    labels : numpy.array
        node labels
    num_samples: int
        the number of samples of each class for training
    random_state : int or None
        random random_state

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    """
    if random_state is not None:
        np.random.seed(random_state)

    idx = np.arange(len(labels))
    nclass = labels.max() + 1
    idx_train = []
    idx_unlabeled = []
    for i in range(nclass):
        labels_i = idx[labels == i]
        labels_i = np.random.permutation(labels_i)
        idx_train.append(labels_i[:num_samples])
        idx_unlabeled.append(labels_i[num_samples:])

    idx_train = np.hstack(idx_train)
    idx_unlabeled = np.hstack(idx_unlabeled)
    idx_unlabeled = np.random.permutation(idx_unlabeled)
    idx_val = idx_unlabeled[: 500]
    idx_test = idx_unlabeled[500: 1500]
    return idx_train, idx_val, idx_test

def get_train_val_test_split(stratify,
                             trainum_examples_per_class: int,
                             val_examples_per_class: int,
                             test_examples_per_class: Optional[None] = None,
                             random_state: Optional[None] = None):

    random_state = np.random.RandomState(random_state)
    remaining_indices = list(range(stratify.shape[0]))

    idx_train = sample_per_class(stratify, trainum_examples_per_class,
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


def sample_per_class(stratify,
                     num_examples_per_class: int,
                     forbidden_indices=None,
                     random_state: Optional[int] = None):

    num_node_classes = stratify.max() + 1
    num_samples = stratify.shape[0]
    sample_indices_per_class = {index: [] for index in range(num_node_classes)}

    # get indices sorted by class
    for class_index in range(num_node_classes):
        for sample_index in range(num_samples):
            if stratify[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def parse_index_file(filename: str) -> List:
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def process_planetoid_datasets(name: str, paths: List[str]) -> Tuple:
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

    node_attr = sp.vstack((allx, tx)).tolil()
    node_attr[test_idx_reorder, :] = node_attr[test_idx_range, :]

    adj_matrix = nx.adjacency_matrix(nx.from_dict_of_lists(
        graph, create_using=nx.DiGraph()))

    node_label = np.vstack((ally, ty))
    node_label[test_idx_reorder, :] = node_label[test_idx_range, :]

    idx_train = np.arange(len(y))
    idx_val = np.arange(len(y), len(y) + 500)
    idx_test = test_idx_range

    node_label = node_label.argmax(1)

    adj_matrix = adj_matrix.astype('float32')
    node_attr = node_attr.astype('float32')

    return adj_matrix, node_attr, node_label, idx_train, idx_val, idx_test
