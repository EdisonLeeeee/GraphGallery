import numpy as np

import gensim
from numba import njit
from gensim.models import Word2Vec
from distutils.version import LooseVersion

from .sklearn_model import SklearnModel


class Deepwalk(SklearnModel):
    """
        Implementation of DeepWalk Unsupervised Graph Neural Networks (DeepWalk). 
        `DeepWalk: Online Learning of Social Representations <https://arxiv.org/abs/1403.6652>`
        Implementation: <https://github.com/phanein/deepwalk>
    """

    def __init__(self, graph, device="cpu", seed=None, name=None, **kwargs):
        r"""Create an unsupervised Deepwalk model.

        This can be instantiated in the following way:

            model = Deepwalk(graph)
                with a graphgallery Graph instance representing
                A sparse, attributed, labeled graph.

        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph`.
            A sparse, labeled graph.
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

    def build(self,
              walk_length=80,
              walks_per_node=10,
              embedding_dim=64,
              window_size=5,
              workers=16,
              epochs=1,
              num_neg_samples=1):
        super().build()
        graph = self.transform.graph_transform(self.graph)
        adj_matrix = graph.adj_matrix
        walks = self.deepwalk_random_walk(adj_matrix.indices,
                                          adj_matrix.indptr,
                                          walk_length=walk_length,
                                          walks_per_node=walks_per_node)

        sentences = [list(map(str, walk)) for walk in walks]
        if LooseVersion(gensim.__version__) <= LooseVersion("4.0.0"):
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
    @njit
    def deepwalk_random_walk(indices,
                             indptr,
                             walk_length=80,
                             walks_per_node=10):

        N = len(indptr) - 1

        for _ in range(walks_per_node):
            for n in range(N):
                single_walk = [n]
                current_node = n
                for _ in range(walk_length - 1):
                    neighbors = indices[
                        indptr[current_node]:indptr[current_node + 1]]
                    if neighbors.size == 0:
                        break
                    current_node = np.random.choice(neighbors)
                    single_walk.append(current_node)

                yield single_walk

    def get_embeddings(self, norm=True):
        if LooseVersion(gensim.__version__) <= LooseVersion("4.0.0"):
            embeddings = self.model.wv.vectors[np.fromiter(
                map(int, self.model.wv.index2word), np.int32).argsort()]
        else:
            embeddings = self.model.wv.vectors[np.fromiter(
                map(int, self.model.wv.index_to_key), np.int32).argsort()]

        if norm:
            embeddings = self.normalize_embedding(embeddings)

        return embeddings
