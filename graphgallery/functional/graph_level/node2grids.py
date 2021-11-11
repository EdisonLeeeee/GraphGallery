import numpy as np
from tqdm import tqdm


class Node2GridsMapper:
    def __init__(self, adj_matrix, feat, biasfactor=0.4, mapsize_a=12, mapsize_b=1):
        self.feat = feat
        self.biasfactor = biasfactor
        self.degree = adj_matrix.sum(1).A1
        self.indices = adj_matrix.indices
        self.indptr = adj_matrix.indptr
        self.K = mapsize_a * mapsize_b
        self.mapsize_a = mapsize_a
        self.mapsize_b = mapsize_b

    def map_node(self, node_index):
        indices = self.indices
        indptr = self.indptr
        K = self.K
        feat = self.feat
        degree = self.degree
        mapsize_a = self.mapsize_a
        mapsize_b = self.mapsize_b
        biasfactor = self.biasfactor

        grids = []
        for node in tqdm(node_index, desc='Converting Node to Grids...'):
            r1_node = indices[indptr[node]:indptr[node + 1]]
            r1_degrees = degree[r1_node]
            sortidx = r1_degrees.argsort()[::-1]
            r1_node = r1_node[sortidx]
            if len(r1_node) < K:
                # find r2 node
                r2_node = set()
                for candidate in r1_node:
                    r2_node.update(indices[indptr[candidate]:indptr[candidate + 1]])
                # difference between r2 and r1
                if r2_node:
                    r2_node = r2_node - set(list(r1_node))
                    if node in r2_node:
                        r2_node.remove(node)
                if r2_node:
                    r2_node = np.array(list(r2_node))
                    r2_degrees = degree[r2_node]
                    sortidx = r2_degrees.argsort()[::-1][:K - len(r1_node)]
                    r2_node = r2_node[sortidx]
                    sampled = np.hstack([r1_node, r2_node])
                else:
                    sampled = r1_node
            else:
                sampled = r1_node[:K]
            # node to grids
            grid = (1 - biasfactor) * feat[sampled] + biasfactor * feat[node]
            delta = K - grid.shape[0]
            if delta > 0:
                grid = np.vstack([grid, np.zeros([delta, grid.shape[-1]])])
            grid = grid.reshape(mapsize_a, mapsize_b, -1)
            grids.append(grid)
        return np.stack(grids, axis=0)
