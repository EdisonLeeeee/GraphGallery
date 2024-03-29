import torch
import numpy as np
import scipy.sparse as sp
import graphgallery as gg
from graphgallery import functional as gf
from torch.utils.data import DataLoader
from functools import partial


__all__ = ["Sequence", "FullBatchSequence", "NullSequence", "NodeSequence", "FastGCNBatchSequence", "NodeLabelSequence", "SAGESequence", "PyGSAGESequence", "SBVATSampleSequence", "MiniBatchSequence", "FeatureLabelSequence"]


def tolist(array):
    if isinstance(array, np.ndarray):
        return array.tolist()
    elif torch.is_tensor(array):
        return array.cpu().tolist()
    else:
        raise ValueError(f"Unable to convert type {type(array)} to list.")


class Sequence(DataLoader):

    def __init__(self, dataset, device='cpu', escape=None, **kwargs):
        super().__init__(dataset, **kwargs)
        self.astensor = partial(gf.astensor, device=device, escape=escape)
        self.astensors = partial(gf.astensors, device=device, escape=escape)
        self.device = device
        self.backend = gg.backend()

    def on_epoch_begin(self):
        ...

    def on_epoch_end(self):
        ...


class FullBatchSequence(Sequence):

    def __init__(self, inputs, y=None, out_index=None, device='cpu', escape=None, **kwargs):
        dataset = gf.astensors(inputs, y, out_index, device=device, escape=escape)
        super().__init__([dataset], batch_size=None, collate_fn=lambda feat: feat, device=device, escape=escape, **kwargs)


class NullSequence(Sequence):

    def __init__(self, *dataset, **kwargs):
        super().__init__([dataset], batch_size=None, collate_fn=lambda feat: feat, **kwargs)


class NodeSequence(Sequence):
    def __init__(self, nodes, **kwargs):
        super().__init__(tolist(nodes), **kwargs)


class NodeLabelSequence(Sequence):
    def __init__(self, nodes, y, **kwargs):
        super().__init__(list(range(len(nodes))), collate_fn=self.collate_fn, **kwargs)
        self.nodes = nodes
        self.y = y

    def collate_fn(self, ids):
        return self.astensors(self.nodes[ids], self.y[ids])


class FeatureLabelSequence(Sequence):
    def __init__(self, feat, y, **kwargs):
        super().__init__(list(range(len(feat))), collate_fn=self.collate_fn, **kwargs)
        self.feat = self.astensor(feat)
        self.y = self.astensor(y)

    def collate_fn(self, ids):
        return self.astensors(self.feat[ids], self.y[ids])


class FastGCNBatchSequence(Sequence):

    def __init__(
        self,
        inputs,
        nodes,
        y=None,
        batch_size=256,
        num_samples=100,
        **kwargs
    ):
        if batch_size is not None:
            super().__init__(list(range(len(nodes))), collate_fn=self.collate_fn, batch_size=batch_size, **kwargs)
        else:
            super().__init__([list(range(len(nodes)))], collate_fn=self.collate_fn, batch_size=batch_size, **kwargs)

        # feat: node feature matrix, which could be numpy array or tensor
        # adj_matrix: node adjacency matrix, which could only be scipy sparse matrix
        self.feat, self.adj_matrix = inputs
        assert sp.isspmatrix(self.adj_matrix), "node adjacency matrix could only be scipy sparse matrix"
        self.y = y
        self.num_samples = num_samples

        if num_samples is not None:
            norm_adjacency = sp.linalg.norm(self.adj_matrix, axis=0)
            self.probability = norm_adjacency / np.sum(norm_adjacency)
        elif batch_size is not None:
            raise RuntimeError("batch training does not work when `num_samples` is None.")

    def collate_fn(self, nodes):
        if self.num_samples is not None:
            sampled_feat, sampled_adjacency, sampled_y = self.sampling(nodes, self.num_samples)
            return self.astensors((sampled_feat, sampled_adjacency), sampled_y)
        else:
            return self.astensors((self.feat, self.adj_matrix), self.y)

    def sampling(self, nodes, num_samples):

        adj_matrix = self.adj_matrix[nodes, :]

        # calculate importance
        neighbors = adj_matrix.sum(0).A1.nonzero()[0]
        if neighbors.size > num_samples:
            probability = self.probability[neighbors]
            probability = probability / np.sum(probability)

            # importance sampling
            sampled_nodes = np.random.choice(neighbors,
                                             size=num_samples,
                                             replace=False,
                                             p=probability
                                             )
        else:
            sampled_nodes = neighbors

        sampled_adjacency = adj_matrix[:, sampled_nodes]
        # normalize
        sampled_probability = self.probability[sampled_nodes]
        sampled_adjacency = sampled_adjacency.dot(sp.diags(
            1.0 / (sampled_probability * num_samples)
        ))
        sampled_feat = self.feat[sampled_nodes]
        sampled_y = self.y[nodes] if self.y is not None else None
        return sampled_feat, sampled_adjacency, sampled_y


class SAGESequence(Sequence):

    def __init__(
        self,
        inputs,
        nodes,
        y=None,
        sizes=[5, 5],
        **kwargs
    ):
        super().__init__(list(range(len(nodes))), collate_fn=self.collate_fn, **kwargs)
        feat, adj_matrix = inputs
        self.feat = self.astensor(feat)
        self.nodes, self.y = nodes, y
        self.sizes = sizes
        self.neighbor_sampler = gg.data.NeighborSampler(adj_matrix)

    def collate_fn(self, ids):
        nodes = self.nodes[ids]
        neighbors = [nodes]

        for size in self.sizes:
            _, nbrs = self.neighbor_sampler.sample(nodes, size=size)
            neighbors.append(nbrs)
            nodes = nbrs

        y = self.y[ids] if self.y is not None else None

        # (node feature matrix, root nodes, 1st-order neighbor, 2nd-order neighbor, ...), node labels
        return (self.feat, *self.astensors(neighbors)), self.astensor(y)


class PyGSAGESequence(Sequence):

    def __init__(
        self,
        inputs,
        nodes,
        y=None,
        sizes=[5, 5],
        **kwargs
    ):
        super().__init__(list(range(len(nodes))), collate_fn=self.collate_fn, **kwargs)
        feat, adj_matrix = inputs
        self.feat = self.astensor(feat)
        self.nodes, self.y = nodes, y
        self.sizes = sizes
        edge_index = torch.LongTensor(adj_matrix.nonzero())
        self.neighbor_sampler = gg.data.PyGNeighborSampler(edge_index, adj_matrix.shape[0])

    def collate_fn(self, ids):
        nodes = torch.LongTensor(self.nodes[ids])
        (batch_size, n_id, adjs) = self.neighbor_sampler.sample(nodes, sizes=self.sizes)

        y = self.y[ids] if self.y is not None else None

        # (node feature matrix, 1st-order adj, 2nd-order adj, ...), node labels
        return (self.feat[n_id], adjs), self.astensor(y)


class SBVATSampleSequence(Sequence):

    def __init__(self, inputs, neighbors, y=None, out_index=None, sizes=50, device='cpu', escape=None, **kwargs):
        dataset = gf.astensors(inputs, y, out_index, device=device, escape=escape)
        super().__init__([dataset], batch_size=None, collate_fn=self.collate_fn, **kwargs)

        self.neighbors = neighbors
        self.num_nodes = inputs[0].shape[0]
        self.sizes = sizes

    def collate_fn(self, dataset):
        adv_mask = self.astensor(self.sample_nodes())
        # ((node feature matrix, adjacency matrix, adv_mask), node labels, out_index)
        dataset = ((*dataset[0], adv_mask), *dataset[1:])
        return dataset

    def sample_nodes(self):
        N = self.num_nodes
        flag = np.zeros(N, dtype=np.bool)
        adv_index = np.zeros(self.sizes, dtype='int32')
        for i in range(self.sizes):
            n = np.random.randint(0, N)
            while flag[n]:
                n = np.random.randint(0, N)
            adv_index[i] = n
            flag[self.neighbors[n]] = True
            if flag.sum() == N:
                break
        adv_mask = np.zeros(N, dtype='float32')
        adv_mask[adv_index] = 1.
        return adv_mask


class MiniBatchSequence(Sequence):
    """Mini-batch sequence used for Cluster-GCN training.

    Parameters
    ----------
    inputs : a list of objects, such as an attributed graph denoted by [ (x1, adj1), (x2, adj2), ...]
    y: if not None, it should be a list of node labels, such as [y1, y2, ...]
    out_index: if not None, it should be a list of node index or mask, such as [index1, index2, ...]
    """

    def __init__(
        self,
        inputs,
        y=None,
        out_index=None,
        node_ids=None,
        **kwargs
    ):

        super().__init__(list(range(len(inputs))), collate_fn=self.collate_fn, batch_size=1, **kwargs)
        self.inputs, self.y, self.out_index, self.node_ids = self.astensors(inputs, y, out_index, node_ids)

    def collate_fn(self, index):
        index = index[0]
        inputs = self.inputs[index]
        y = self.y[index] if self.y is not None else None
        out_index = self.out_index[index] if self.out_index is not None else None
        node_ids = self.node_ids[index] if self.node_ids is not None else None
        if node_ids is None:
            return inputs, y, out_index
        else:
            return inputs, y, out_index, node_ids
