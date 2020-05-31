import numpy as np
import networkx as nx

from gensim.models import Word2Vec

from graphgallery.nn.models import UnsupervisedModel
from graphgallery.utils.walker import RandomWalker, alias_sample


class Node2vec(UnsupervisedModel):
    """
        Implementation of Node2vec Unsupervised Graph Neural Networks (Node2vec). 
        [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653)
        Implementation: https://github.com/aditya-grover/node2vec
        Cpp implementation: https://github.com/snap-stanford/snap/tree/master/examples/node2vec

        Arguments:
        ----------
            adj: `scipy.sparse.csr_matrix` (or `csc_matrix`) with shape (N, N)
                The input `symmetric` adjacency matrix, where `N` is the number of nodes 
                in graph.
            features: `np.array` with shape (N, F)
                The input node feature matrix, where `F` is the dimension of node features.
            labels: `np.array` with shape (N,)
                The ground-truth labels for all nodes in graph.
            device (String, optional): 
                The device where the model is running on. You can specified `CPU` or `GPU` 
                for the model. (default: :obj: `CPU:0`, i.e., the model is running on 
                the 0-th device `CPU`)
            seed (Positive integer, optional): 
                Used in combination with `tf.random.set_seed & np.random.seed & random.seed` 
                to create a reproducible sequence of tensors across multiple calls. 
                (default :obj: `None`, i.e., using random seed)
            name (String, optional): 
                Name for the model. (default: name of class)

    """

    def __init__(self, adj, features, labels, graph=None, device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, features, labels, device=device, seed=seed, name=name, **kwargs)


        if graph is None:
            graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph)

        self.walker = None
        self.graph = graph

    def build(self, walk_length=80, walks_per_node=10,
              embedding_dim=64, window_size=5, workers=16,
              iter=1, num_neg_samples=1, p=0.5, q=0.5):

        self.walker = RandomWalker(self.graph, p=p, q=q)
        self.walker.preprocess_transition_probs()

        walks = self.node2vec_random_walk(self.graph,
                                          self.walker.alias_nodes,
                                          self.walker.alias_edges,
                                          walk_length=walk_length,
                                          walks_per_node=walks_per_node)

        sentences = [list(map(str, walk)) for walk in walks]

        model = Word2Vec(sentences, size=embedding_dim, window=window_size, min_count=0, sg=1, workers=workers,
                         iter=iter, negative=num_neg_samples, hs=0, compute_loss=True)
        
        self.set_model(model)

    @staticmethod
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

    def get_embeddings(self, norm=True):
        embeddings = self.model.wv.vectors[np.fromiter(map(int, self.model.wv.index2word), np.int32).argsort()]

        if norm:
            embeddings = self.normalize_embedding(embeddings)

        self.embeddings = embeddings
