from functools import partial

import torch
import numpy as np
import scipy.sparse as sp
import graphgallery as gg
from graphgallery import functional as gf
from torch.utils.data import DataLoader


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
        super().__init__([dataset], batch_size=None, collate_fn=lambda x: x, device=device, escape=escape, **kwargs)


class NullSequence(Sequence):

    def __init__(self, *dataset, **kwargs):
        super().__init__([dataset], batch_size=None, collate_fn=lambda x: x, **kwargs)


def tolist(array):
    if isinstance(array, np.ndarray):
        return array.tolist()
    elif torch.is_tensor(array):
        return array.cpu().tolist()
    else:
        raise ValueError(f"Unable to convert type {type(array)} to list.")


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
            super().__init__(tolist(nodes), collate_fn=self.collate_fn, batch_size=batch_size, **kwargs)
        else:
            super().__init__([nodes], collate_fn=self.collate_fn, batch_size=batch_size, **kwargs)

        # x: node attribute matrix, which could be numpy array or tensor
        # adj_matrix: node adjacency matrix, which could only be scipy sparse matrix
        self.x, self.adj_matrix = inputs
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
            sampled_x, sampled_adjacency, sampled_y = self.sampling(nodes, self.num_samples)
            return self.astensors((sampled_x, sampled_adjacency), sampled_y)
        else:
            return self.astensors((self.x, self.adj_matrix), self.y)

    def sampling(self, nodes, num_samples):

        # 采样的源节点的邻接矩阵, 使用初始邻接矩阵相应的行
        adj_matrix = self.adj_matrix[nodes, :]

        # 对源节点所有可用的目标节点计算重要性
        neighbors = adj_matrix.sum(0).A1.nonzero()[0]
        if neighbors.size > num_samples:
            probability = self.probability[neighbors]
            probability = probability / np.sum(probability)

            # 对目标节点按重要性采样
            sampled_nodes = np.random.choice(neighbors,
                                             size=num_samples,
                                             replace=False,
                                             p=probability
                                             )
        else:
            sampled_nodes = neighbors

        # 获得采样后的由源节点和目标节点组成的邻接矩阵
        sampled_adjacency = adj_matrix[:, sampled_nodes]
        # 邻接矩阵归一化
        sampled_probability = self.probability[sampled_nodes]
        sampled_adjacency = sampled_adjacency.dot(sp.diags(
            1.0 / (sampled_probability * num_samples)
        ))
        # FIXME: Only integers, slices (`:`), ellipsis (`...`), tf.newaxis (`None`) and scalar tf.int32/tf.int64 tensors are valid indices for tensorflow tensor
        # So the following codes would triggered error for tensorflow backend
        # sampled_x = self.x[sampled_nodes]
        # sampled_y = self.y[nodes] if self.y is not None else None
        sampled_x = gf.gather(self.x, sampled_nodes)
        sampled_y = gf.gather(self.y, nodes)
        return sampled_x, sampled_adjacency, sampled_y


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
        x, adj_matrix = inputs
        self.x = self.astensor(x)
        self.nodes, self.y = nodes, y
        self.sizes = sizes
        self.neighbor_sampler = gg.utils.NeighborSampler(adj_matrix)
        self.is_tensorflow_backend = self.backend == "tensorflow"

    def collate_fn(self, ids):
        is_tensorflow_backend = self.is_tensorflow_backend
        nodes = self.nodes[ids]
        neighbors = [nodes]

        for size in self.sizes:
            _, nbrs = self.neighbor_sampler.sample(nodes, size=size, as_numpy=is_tensorflow_backend)
            neighbors.append(nbrs)
            nodes = nbrs

        y = self.y[ids] if self.y is not None else None

        # (node attribute matrix, root nodes, 1st-order neighbor, 2nd-order neighbor, ...), node labels
        return (self.x, *self.astensors(neighbors)), self.astensor(y)
