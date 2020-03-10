from gensim.models import Word2Vec
from numba import njit
import numpy as np

@njit
def deepwalk_random_walk(indices, indptr, walk_length=80, walks_per_node=10):
    
    N = len(indptr) - 1

    for _ in range(walks_per_node):
        for n in range(N):
            single_walk = [n]
            current_node = n
            for _ in range(walk_length-1):
                neighbors = indices[indptr[current_node]:indptr[current_node + 1]]
                if neighbors.size == 0: break
                current_node = np.random.choice(neighbors)
                single_walk.append(current_node)
                
            yield single_walk



class DeepWalk:

    def __init__(self, adj, walk_length=80, walks_per_node=10):

        self.model = None
        self.embeddings = None

        walks = deepwalk_random_walk(adj.indices, 
                                     adj.indptr,
                                     walk_length=walk_length,
                                     walks_per_node=walks_per_node)
        

        self.sentences = [list(map(str, walk)) for walk in walks]
        
#     @staticmethod
#     def deepwalk_random_walk(adj, walk_length, walks_per_node):
        
#         import torch
#         import torch_cluster

#         edge_index = torch.LongTensor(adj.nonzero())
#         start = torch.arange(adj.shape[0]).repeat(walks_per_node)
        
#         row, col = edge_index
#         walks = torch_cluster.random_walk(row, col, start, walk_length)

#         return walks.numpy()

    def train(self, embedding_dim=128, window_size=5, workers=16, iter=1, num_neg_samples=1):

        self.model = Word2Vec(self.sentences, size=embedding_dim, window=window_size, min_count=0, sg=1, workers=workers,
                     iter=iter, negative=num_neg_samples, hs=0, compute_loss=True)


    def get_embeddings(self,):
        if self.model is None:
            print("model not train")
            return None

#         self.embeddings = self.model.wv.vectors[np.fromiter(map(int, self.model.wv.index2word), np.int32).argsort()]
        self.embeddings = self.model.wv.syn0[np.fromiter(map(int, self.model.wv.index2word), np.int32).argsort()]

        return self.embeddings
    
    

