import numpy as np
import scipy.sparse as sp

from graphgallery import functional as gf
from .attacker import Attacker


class FlipAttacker(Attacker):
    def __init__(self, graph, device="cpu", seed=None, name=None, **kwargs):
        super().__init__(graph, device=device, seed=seed, name=name, **kwargs)
        assert not graph.is_multiple(), "NOT Supported for multiple graph"
        self.nattr_flips = None
        self.eattr_flips = None
        self.adj_flips = None

    @property
    def A(self):
        adj_flips = self.edge_flips
        if self.modified_adj is None:
            if adj_flips is not None:
                self.modified_adj = gf.flip_adj(self.graph.adj_matrix,
                                                adj_flips)
            else:
                self.modified_adj = self.graph.adj_matrix

        adj = self.modified_adj

        if gf.is_anytensor(adj):
            adj = gf.tensoras(adj)

        if isinstance(adj, np.ndarray):
            adj = sp.csr_matrix(adj)
        elif sp.isspmatrix(adj):
            adj = adj.tocsr(copy=False)
        else:
            raise TypeError(adj)

        return adj

    @property
    def x(self):
        return self.nx

    @property
    def nx(self):
        attr_flips = self.nx_flips
        if self.modified_nx is None:
            if attr_flips is not None:
                self.modified_nx = gf.flip_attr(self.graph.node_attr,
                                                attr_flips)
            else:
                self.modified_nx = self.graph.node_attr

        x = self.modified_nx

        if sp.isspmatrix(x):
            x = x.A
        elif gf.is_anytensor(x):
            x = gf.tensoras(x)
        elif not isinstance(x, np.ndarray):
            raise TypeError(x)
        return x

    @property
    def ex(self):
        # TODO
        return None

    @property
    def d(self):
        if self.modified_degree is None:
            self.modified_degree = self.A.sum(1).A1.astype(self.intx)
        return self.modified_degree

    @property
    def edge_flips(self):
        flips = self.adj_flips
        if flips is None or len(flips) == 0:
            return None

        if isinstance(flips, dict):
            flips = list(flips.keys())

        return np.asarray(flips, dtype="int64")

    @property
    def nx_flips(self):
        flips = self.nattr_flips
        if flips is None or len(flips) == 0:
            return None

        if isinstance(flips, dict):
            flips = list(flips.keys())

        return np.asarray(flips, dtype="int64")

    @property
    def ex_flips(self):
        # TODO
        return None

    @property
    def flips(self):
        # TODO
        return gf.BunchDict(edge_flips=self.edge_flips, nx_flips=self.nx_flips)

    def show(self):
        flips = self.edge_flips

        if flips is not None and len(flips) != 0:
            row, col = flips.T
            w = self.graph.adj_matrix[row, col].A1
            added = (w == 0)
            removed = (w > 0)

            print(f"Flip {flips.shape[0]} edges, where {added.sum()} are added, {removed.sum()} are removed.")
            diff = self.graph.node_label[row[added]] != self.graph.node_label[col[added]]
            ratio = diff.sum() / diff.size if diff.size else 0.
            print(f"For added edges, {ratio:.2%} of them belong to different classes.")
            same = self.graph.node_label[row[removed]] == self.graph.node_label[col[removed]]
            ratio = same.sum() / same.size if same.size else 0.
            print(f"For removed edges, {ratio:.2%} of them belong to the same class.")
        else:
            print("No edge flips found.")

        # TODO: nattr_flips
