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


class Node2VecE(Trainer):
    r"""An implementation of `"Node2Vec+" <https://arxiv.org/abs/2109.08031>`_
    from the arxiv'21 paper "Accurately Modeling Biased Random Walks on Weighted Wraphs Using Node2vec+". `E` means the `extend` verison of node2vec. if `extend=False`, it is equivalent to standard `node2vec` but is faster particularly on larger graphs.


    Parameters:
    -----------
    walk_number (int): Number of random walks. Default is 10.
    walk_length (int): Length of random walks. Default is 80.
    p (float): Return parameter (1/p transition probability) to move towards from previous node.
    q (float): In-out parameter (1/q transition probability) to move away from previous node.
    extend (bool): whether to use the extended version (`node2vec`). See below.    
    mode (str): different modes including `PreComp` and `SparseOTF`. See below.


    Specify `extend=True` for using node2vec+, which is a natural extension of
    node2vec and handles weighted graph more effectively. For more information, see
    `Accurately Modeling Biased Random Walks on Weighted Wraphs Using Node2vec+`(https://arxiv.org/abs/2109.08031)


    `Node2VecE` operates in three different modes – PreComp and SparseOTF – that are optimized for networks of different sizes and densities:
    - `PreComp` for networks that are small (≤10k nodes; any density),
    - `SparseOTF` for networks that are large and sparse (>10k nodes; ≤10% of edges),
    These modes appropriately take advantage of compact/dense graph data structures, precomputing transition probabilities, and computing 2nd-order transition probabilities during walk generation to achieve significant improvements in performance. 
    """

    def __init__(self, walk_length: int = 80, walk_number: int = 10,
                 p: float = 0.5, q: float = 0.5, dimensions: int = 64,
                 workers: int = 3, window_size: int = 5, epochs: int = 1,
                 learning_rate: float = 0.025, negative: int = 1, extend=True,
                 mode='PreComp',
                 name: str = None,
                 seed: int = None):
        kwargs = locals()
        kwargs.pop("self")
        super().__init__(**kwargs)

    def fit_step(self, graph: sp.csr_matrix):
        sentences = BiasedRandomWalkerAlias(walk_length=self.walk_length,
                                            walk_number=self.walk_number,
                                            p=self.p, q=self.q, extend=self.extend,
                                            mode=self.mode).walk(graph)
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
