import numpy as np
import scipy.sparse as sp
from graphgallery import functional as gf
from sklearn.preprocessing import LabelEncoder

from .io import read_csv, read_json


class Reader:
    @staticmethod
    def read_graphs(filepath):
        graphs = read_json(filepath)
        graphs = [gf.edge_to_sparse_adj(graphs[str(i)]) for i in range(len(graphs))]
        return graphs

    @staticmethod
    def read_edges(filepath, src='id_1', dst='id_2'):
        data = read_csv(filepath)
        row = data[src].to_numpy()
        col = data[dst].to_numpy()
        N = max(row.max(), col.max()) + 1
        graph = sp.csr_matrix((np.ones(row.shape[0], dtype=np.float32), (row, col)), shape=(N, N))
        return graph

    @staticmethod
    def read_csv_features(filepath):
        data = read_csv(filepath)
        row = np.array(data["node_id"])
        col = np.array(data["feature_id"])
        values = np.array(data["value"])
        node_count = max(row) + 1
        feature_count = max(col) + 1
        shape = (node_count, feature_count)
        features = sp.csr_matrix((values, (row, col)), shape=shape)
        return features

    @staticmethod
    def read_json_features(filepath):
        data = read_json(filepath)
        rows = []
        cols = []
        for k, v in data.items():
            k = int(k)
            rows += [k] * len(v)
            cols += v

        N = max(rows) + 1
        M = max(cols) + 1
        features = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(N, M))
        return features

    @staticmethod
    def read_target(filepath, return_target=True):
        data = read_csv(filepath)
        if return_target:
            return data["target"].to_numpy()
        else:
            return data
