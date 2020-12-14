import numpy as np
import networkx as nx

import gensim

from gensim.models import Word2Vec
from distutils.version import LooseVersion

from graphgallery.utils.walker import RandomWalker, alias_sample
from .sklearn_model import SklearnModel



class Node2vec(SklearnModel):
    """
        Implementation of Node2vec Unsupervised Graph Neural Networks (Node2vec). 
        `node2vec: Scalable attribute Learning for Networks <https://arxiv.org/abs/1607.00653>`
        Implementation: <https://github.com/aditya-grover/node2vec>
        Cpp implementation: <https://github.com/snap-stanford/snap/tree/master/examples/node2vec>

    """

    def __init__(self, graph, device="cpu", seed=None, name=None, **kwargs):
        r"""Create an unsupervised Node2Vec model.

        This can be instantiated in the following way:

            model = Node2vec(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph`.
            A sparse, labeled graph.
        graph_transform: string, `transform` or None. optional
            How to transform the graph, by default None.
        device: string. optional
            The device where the model is running on. 
            You can specified ``CPU``, ``GPU`` or ``cuda`` 
            for the model. (default: :str: `cpu`, i.e., running on the `CPU`)
        seed: interger scalar. optional 
            Used in combination with `tf.random.set_seed` & `np.random.seed` 
            & `random.seed` to create a reproducible sequence of tensors across 
            multiple calls. (default :obj: `None`, i.e., using random seed)
        name: string. optional
            Specified name for the model. (default: :str: `class.__name__`)        
        kwargs: keyword parameters for transform, including:
            ``adj_transform``, ``attr_transform``, 
            ``label_transform``, ``graph_transform``, etc.
        """
        super().__init__(graph, device=device, seed=seed, name=name, **kwargs)

        self.nxgraph = self.graph.nxgraph()

    def build(self,
              walk_length=80,
              walks_per_node=10,
              embedding_dim=64,
              window_size=5,
              workers=16,
              epochs=1,
              num_neg_samples=1,
              p=0.5,
              q=0.5):
        super().build()

        self.walker = RandomWalker(self.nxgraph, p=p, q=q)
        self.walker.preprocess_transition_probs()

        walks = self.node2vec_random_walk(self.nxgraph,
                                          self.walker.alias_nodes,
                                          self.walker.alias_edges,
                                          walk_length=walk_length,
                                          walks_per_node=walks_per_node)

        sentences = [list(map(str, walk)) for walk in walks]
        if LooseVersion(gensim.__version__)<=LooseVersion("4.0.0"):
             model = Word2Vec(sentences,
                             size=embedding_dim,
                             window=window_size,
                             min_count=0,
                             sg=1,
                             workers=workers,
                             iter=epochs,
                             negative=num_neg_samples,
                             hs=0,
                             compute_loss=True)
           
        else:
            model = Word2Vec(sentences,
                             vector_size=embedding_dim,
                             window=window_size,
                             min_count=0,
                             sg=1,
                             workers=workers,
                             epochs=epochs,
                             negative=num_neg_samples,
                             hs=0,
                             compute_loss=True)
            
        self.model = model

    @staticmethod
    def node2vec_random_walk(G,
                             alias_nodes,
                             alias_edges,
                             walk_length=80,
                             walks_per_node=10):
        for _ in range(walks_per_node):
            for n in G.nodes():
                single_walk = [n]
                current_node = n
                for _ in range(walk_length - 1):
                    neighbors = list(G.neighbors(current_node))
                    if len(neighbors) > 0:
                        if len(single_walk) == 1:
                            current_node = neighbors[alias_sample(
                                alias_nodes[current_node][0],
                                alias_nodes[current_node][1])]
                        else:
                            prev = single_walk[-2]
                            edge = (prev, current_node)
                            current_node = neighbors[alias_sample(
                                alias_edges[edge][0], alias_edges[edge][1])]
                    else:
                        break

                    single_walk.append(current_node)
                yield single_walk

    def get_embeddings(self, norm=True):
        if LooseVersion(gensim.__version__)<=LooseVersion("4.0.0"):
            embeddings = self.model.wv.vectors[np.fromiter(
                map(int, self.model.wv.index2word), np.int32).argsort()]
        else:
            embeddings = self.model.wv.vectors[np.fromiter(
                map(int, self.model.wv.index_to_key), np.int32).argsort()]

        if norm:
            embeddings = self.normalize_embedding(embeddings)

        return embeddings
