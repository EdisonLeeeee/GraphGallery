import os
import errno

import networkx as nx
import scipy.sparse as sp
import pickle as pkl
import os.path as osp
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split



def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e
        
def download_file(raw_paths, urls):
    
    last_except = None
    for file_name, url in zip(raw_paths, urls):
        try:
            tf.keras.utils.get_file(file_name, origin=url)
        except Exception as e:
            last_except = e
            print(e)

    if last_except is not None:
        raise last_except

def files_exist(files):
    return len(files) != 0 and all([osp.exists(f) for f in files])


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



def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def process_planetoid_datasets(name, paths):
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
        test_idx_range_full = np.arange(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended


    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.toarray()

    graph = nx.from_dict_of_lists(graph, create_using=nx.DiGraph())
    adj = nx.adjacency_matrix(graph)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]


    idx_train = np.arange(len(y))
    idx_val = np.arange(len(y), len(y)+500)
    idx_test = test_idx_range

    labels = labels.argmax(1)
    
    adj = adj.astype('float32')
    return adj, features, labels, idx_train, idx_val, idx_test


