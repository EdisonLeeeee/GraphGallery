import scipy.sparse as sp
from .walker import BiasedRandomWalker, BiasedRandomWalkerAlias
from .utils import Word2Vec
from .trainer import Trainer


class Node2Vec(Trainer):
    r"""An implementation of `"Node2Vec" <https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf>`_
    from the KDD '16 paper "node2vec: Scalable Feature Learning for Networks".
    The procedure uses biased second order random walks to approximate the pointwise mutual information
    matrix obtained by pooling normalized adjacency matrix powers. This matrix
    is decomposed by an approximate factorization technique.

    """

    def __init__(self, walk_length: int = 80, walk_number: int = 10,
                 p: float = 0.5, q: float = 0.5, dimensions: int = 64,
                 workers: int = 3, window_size: int = 5, epochs: int = 1,
                 learning_rate: float = 0.025, negative: int = 1,
                 name: str = None,
                 seed: int = None):
        kwargs = locals()
        kwargs.pop("self")
        super().__init__(**kwargs)

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.negative = negative
        self.p = p
        self.q = q

    def fit_step(self, graph: sp.csr_matrix):
        walks = BiasedRandomWalker(walk_length=self.walk_length,
                                   walk_number=self.walk_number,
                                   p=self.p, q=self.q).walk(graph)
        sentences = [list(map(str, walk)) for walk in walks]
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

        self._embedding = model.get_embedding()
