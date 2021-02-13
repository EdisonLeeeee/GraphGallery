import numpy as np
import networkx as nx

import gensim

from gensim.models import Word2Vec
from distutils.version import LooseVersion
from sklearn.linear_model import LogisticRegression

from graphgallery.gallery.nodeclas.utils.walker import RandomWalker, alias_sample
from graphgallery.gallery.nodeclas import Common
from graphgallery import functional as gf
from .sklearn_model import SklearnModel


@Common.register()
class Node2vec(SklearnModel):
    """
        Implementation of Node2vec Unsupervised Graph Neural Networks (Node2vec).
        `node2vec: Scalable attribute Learning for Networks <https://arxiv.org/abs/1607.00653>`
        Implementation: <https://github.com/aditya-grover/node2vec>
        Cpp implementation: <https://github.com/snap-stanford/snap/tree/master/examples/node2vec>

    """

    def process_step(self,
                     adj_transform=None,
                     attr_transform=None,
                     graph_transform=None,
                     p=0.5,
                     q=0.5,
                     walk_length=80,
                     walks_per_node=10):

        graph = gf.get(graph_transform)(self.graph).nxgraph()
        walker = RandomWalker(graph, p=p, q=q)
        walker.preprocess_transition_probs()

        walks = self.node2vec_random_walk(graph,
                                          walker.alias_nodes,
                                          walker.alias_edges,
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

    @property
    def embeddings(self):
        if LooseVersion(gensim.__version__) <= LooseVersion("4.0.0"):
            embeddings = self.model.wv.vectors[np.fromiter(
                map(int, self.model.wv.index2word), np.int32).argsort()]
        else:
            embeddings = self.model.wv.vectors[np.fromiter(
                map(int, self.model.wv.index_to_key), np.int32).argsort()]

        if self.cfg.normalize_embedding:
            embeddings = self.normalize_embedding(embeddings)

        return embeddings
