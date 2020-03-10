from gensim.models import Word2Vec
import numpy as np
import networkx as nx
from .walker import RandomWalker, alias_sample


def node2vec_random_walk(G, alias_nodes, alias_edges, walk_length=80, walks_per_node=10):
    for _ in range(walks_per_node):
        for n in G.nodes():
            single_walk = [n]
            current_node = n
            for _ in range(walk_length-1):
                neighbors = list(G.neighbors(current_node))
                if len(neighbors) > 0:
                    if len(single_walk) == 1:
                        current_node = neighbors[alias_sample(alias_nodes[current_node][0], alias_nodes[current_node][1])]
                    else:
                        prev = single_walk[-2]
                        edge = (prev, current_node)
                        current_node = neighbors[alias_sample(alias_edges[edge][0], alias_edges[edge][1])]
                else:
                    break
                    
                single_walk.append(current_node)
            yield single_walk


class Node2Vec:

    def __init__(self, adj, walk_length=80, walks_per_node=10, p=0.5, q=0.5):

            
        self.model = None
        self.embeddings = None
        
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph)
        
        self.walker = RandomWalker(graph, p=p, q=q)
        self.walker.preprocess_transition_probs()

        walks = node2vec_random_walk(graph,
                                     self.walker.alias_nodes, 
                                     self.walker.alias_edges, 
                                     walk_length=walk_length,
                                     walks_per_node=walks_per_node)
        
        self.sentences = [list(map(str, walk)) for walk in walks]

    def train(self, embedding_dim=128, window_size=5, workers=16, iter=1, num_neg_samples=1):

        self.model = Word2Vec(self.sentences, size=embedding_dim, window=window_size, min_count=0, sg=1, workers=workers,
                     iter=iter, negative=num_neg_samples, hs=0, compute_loss=True)


    def get_embeddings(self,):
        if self.model is None:
            print("model not train")
            return None

        self.embeddings = self.model.wv.vectors[np.fromiter(map(int, self.model.wv.index2word), np.int32).argsort()]
#         self.embeddings = self.model.wv.syn0[np.fromiter(map(int, self.model.wv.index2word), np.int32).argsort()]

        return self.embeddings

