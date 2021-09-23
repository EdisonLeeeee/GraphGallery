import numpy as np
import numba as nb
from tqdm import tqdm


class Node2GridsMapper:
    def __init__(self, adj_matrix, x, biasfactor=0.4, mapsize_a=12, mapsize_b=1):
        self.x = x
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
        x = self.x
        degree = self.degree
        mapsize_a = self.mapsize_a
        mapsize_b = self.mapsize_b
        biasfactor = self.biasfactor

        grids = []
        for node in tqdm(node_index, desc='Converting Node to Grids...'):
            nbrs, distance = self.K_hop_neighbors(indices, indptr, node, hops=2)
            nbrs = np.asarray(nbrs)
            distance = np.asarray(distance)
            first_level = nbrs[distance == 1]
            num_1st_nbrs = np.size(first_level)
            if num_1st_nbrs >= K:
                sorted_idx = degree[first_level].argsort()[-K:]  # sort by degree
                sampled = first_level[sorted_idx]
            else:
                second_level = nbrs[distance == 2]
                num_2rd_nbrs = np.size(second_level)
                sorted_idx = degree[second_level].argsort()[-(K - num_1st_nbrs):]
                # 1st-hop neighbors + 2rd-hop neighbors
                sampled = np.hstack([first_level, second_level[sorted_idx]])

            # node to grids
            grid = (1 - biasfactor) * x[sampled] + biasfactor * x[node]
            delta = K - grid.shape[0]
            if delta > 0:
                grid = np.vstack([grid, np.zeros([delta, grid.shape[-1]])])
            grid = grid.reshape(mapsize_a, mapsize_b, -1)
            grids.append(grid)
        return np.stack(grids, axis=0)

    @staticmethod
    @nb.njit(nogil=True)
    def K_hop_neighbors(indices, indptr, root, hops=1):
        N = len(indptr) - 1
        seen = np.zeros(N) - 1
        seen[root] = 0
        start = 0
        queue = [root]
        distance = [0]
        for level in range(hops):
            end = len(queue)
            while start < end:
                head = queue[start]
                nbrs = indices[indptr[head]:indptr[head + 1]]
                for u in nbrs:
                    if seen[u] < 0:
                        queue.append(u)
                        seen[u] = level + 1
                        distance.append(level + 1)
                start += 1
        return queue, distance
