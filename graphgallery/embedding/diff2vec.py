import numpy as np
import networkx as nx
from gensim.models.word2vec import Word2Vec
from diffuser import EulerianDiffuser
from sklearn import preprocessing


class Diff2Vec:
    r"""An implementation of `"Diff2Vec" <http://homepages.inf.ed.ac.uk/s1668259/papers/sequence.pdf>`_
    from the CompleNet '18 paper "Diff2Vec: Fast Sequence Based Embedding with Diffusion Graphs".
    The procedure creates diffusion trees from every source node in the graph. These graphs are linearized
    by a directed Eulerian walk, the walks are used for running the skip-gram algorithm the learn node
    level neighbourhood based embeddings.

    """

    def __init__(self, diffusion_cover: int=80, diffusion_number: int=10, dimensions: int = 64,
                 workers: int = 16, window_size: int = 5, epochs: int = 1,
                 learning_rate: float = 0.025, negative: int = 5, seed: int = None):

        self.diffusion_cover = diffusion_cover
        self.diffusion_number = diffusion_number
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.negative = negative
        self.seed = seed

    def fit(self, graph):
        diffusions = EulerianDiffuser(diffusion_cover=self.diffusion_cover,
                             diffusion_number=self.diffusion_number).diffusion(graph)
        sentences = [list(map(str, diffusion)) for diffusion in diffusions]
        model = Word2Vec(sentences,
                         sg=1,
                         hs=0,
                         alpha=self.learning_rate,
                         iter=self.epochs,
                         size=self.dimensions,
                         window=self.window_size,
                         workers=self.workers,
                         negative=self.negative,
                         seed=self.seed)

        self._embedding = model.wv.vectors[np.fromiter(map(int, model.wv.index2word), np.int32).argsort()]

    def get_embedding(self, normalize=True) -> np.array:
        """Getting the node embedding."""
        embedding = self._embedding
        if normalize:
            embedding = preprocessing.normalize(embedding)            
        return embedding
