import numpy as np

from gensim.models import Word2Vec
from numba import njit
from .base import UnsupervisedModel


class Deepwalk(UnsupervisedModel):

    def __init__(self, adj, features, labels, device='CPU:0', seed=None):

        super().__init__(adj, features, labels, device=device, seed=seed)


    def build(self, walk_length=80, walks_per_node=10, 
              embedding_dim=64, window_size=5, workers=16, 
              iter=1, num_neg_samples=1):
        
        walks = self.deepwalk_random_walk(self.adj.indices, 
                                     self.adj.indptr,
                                     walk_length=walk_length,
                                     walks_per_node=walks_per_node)
        

        sentences = [list(map(str, walk)) for walk in walks]
        
        model = Word2Vec(sentences, size=embedding_dim, window=window_size, min_count=0, sg=1, workers=workers,
                     iter=iter, negative=num_neg_samples, hs=0, compute_loss=True)
        self.model = model
        
    @staticmethod
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
                
    def get_embeddings(self, norm=True):
        embeddings = self.model.wv.vectors[np.fromiter(map(int, self.model.wv.index2word), np.int32).argsort()]

        if norm:
            embeddings = self._normalize_embedding(embeddings)

        self.embeddings = embeddings
