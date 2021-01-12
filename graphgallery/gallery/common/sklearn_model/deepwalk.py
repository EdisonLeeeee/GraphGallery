import numpy as np

import gensim
from numba import njit
from gensim.models import Word2Vec
from distutils.version import LooseVersion

from sklearn.linear_model import LogisticRegression
from graphgallery import functional as gf
from graphgallery.gallery import Common
from .sklearn_model import SklearnModel


@Common.register()
class Deepwalk(SklearnModel):
    """
        Implementation of DeepWalk Unsupervised Graph Neural Networks (DeepWalk).
        `DeepWalk: Online Learning of Social Representations <https://arxiv.org/abs/1403.6652>`
        Implementation: <https://github.com/phanein/deepwalk>
    """

    def process_step(self,
                     adj_transform=None,
                     attr_transform=None,
                     graph_transform=None,
                     walk_length=80,
                     walks_per_node=10):
        graph = gf.get(graph_transform)(self.graph)
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        walks = self.deepwalk_random_walk(adj_matrix.indices,
                                          adj_matrix.indptr,
                                          walk_length=walk_length,
                                          walks_per_node=walks_per_node)

        self.register_cache(walks=walks)

    def model_builder(self,
                      name="Word2Vec",
                      embedding_dim=64,
                      window_size=5,
                      workers=16,
                      epochs=1,
                      num_neg_samples=1):

        assert name == "Word2Vec"

        walks = self.cache.walks
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
        return model

    def classifier_builder(self):
        cfg = self.cfg.classifier
        assert cfg.name == "LogisticRegression"
        classifier = LogisticRegression(solver=cfg.solver,
                                        max_iter=cfg.max_iter,
                                        multi_class=cfg.multi_class,
                                        random_state=cfg.random_state)
        return classifier

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

    @property
    def embeddings(self, norm=True):
        if LooseVersion(gensim.__version__) <= LooseVersion("4.0.0"):
            embeddings = self.model.wv.vectors[np.fromiter(
                map(int, self.model.wv.index2word), np.int32).argsort()]
        else:
            embeddings = self.model.wv.vectors[np.fromiter(
                map(int, self.model.wv.index_to_key), np.int32).argsort()]

        if self.cfg.normalize_embedding:
            embeddings = self.normalize_embedding(embeddings)

        return embeddings
