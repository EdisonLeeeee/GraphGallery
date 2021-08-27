from .diffuser import EulerianDiffuser
from .utils import Word2Vec
from .trainer import Trainer


class Diff2Vec(Trainer):
    r"""An implementation of `"Diff2Vec" <http://homepages.inf.ed.ac.uk/s1668259/papers/sequence.pdf>`_
    from the CompleNet '18 paper "Diff2Vec: Fast Sequence Based Embedding with Diffusion Graphs".
    The procedure creates diffusion trees from every source node in the graph. These graphs are linearized
    by a directed Eulerian walk, the walks are used for running the skip-gram algorithm the learn node
    level neighbourhood based embeddings.
    """

    def __init__(self, diffusion_cover: int = 80,
                 diffusion_number: int = 10, dimensions: int = 64,
                 workers: int = 3, window_size: int = 5, epochs: int = 1,
                 learning_rate: float = 0.025, negative: int = 5,
                 name: str = None, seed: int = None):
        kwargs = locals()
        kwargs.pop("self")
        super().__init__(**kwargs)
        self.diffusion_cover = diffusion_cover
        self.diffusion_number = diffusion_number
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.negative = negative

    def fit_step(self, graph):
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

        self._embedding = model.get_embedding()
