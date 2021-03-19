import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from treefeatures import WeisfeilerLehmanHashing

from walker import RandomWalker
from sklearn import preprocessing


class Role2Vec:
    r"""An implementation of `"Role2vec" <https://arxiv.org/abs/1802.02896>`_
    from the IJCAI '18 paper "Learning Role-based Graph Embeddings".
    The procedure uses random walks to approximate the pointwise mutual information
    matrix obtained by multiplying the pooled adjacency power matrix with a 
    structural feature matrix (in this case Weisfeiler-Lehman features). This way
    one gets structural node embeddings.
    """

    def __init__(self, walk_number: int = 10, walk_length: int = 80, dimensions: int = 64, workers: int = 16,
                 window_size: int = 5, epochs: int = 1, learning_rate: float = 0.025, down_sampling: float = 0.0001,
                 min_count: int = 10, wl_iterations: int = 2, seed: int = 42):

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.down_sampling = down_sampling
        self.min_count = min_count
        self.wl_iterations = wl_iterations
        self.seed = seed

    def _create_documents(self, walks, features):
        """
        Accumulating the WL feature in neighbourhoods.
        """
        new_features = {node: [] for node, feature in features.items()}
        for walk in walks:
            for i in range(self.walk_length - self.window_size):
                for j in range(self.window_size):
                    source = walk[i]
                    target = walk[i + j]
                    new_features[source].append(features[target])
                    new_features[target].append(features[source])

        new_features = {node: [feature for features in new_features[node] for feature in features] for node, _ in new_features.items()}
        new_features = [TaggedDocument(words=feature, tags=[str(node)]) for node, feature in new_features.items()]
        return new_features

    def fit(self, graph: sp.csr_matrix):
        """
        Fitting a Role2vec model.
        """
        walks = RandomWalker(walk_length=self.walk_length,
                             walk_number=self.walk_number).walk(graph)

        hasher = WeisfeilerLehmanHashing(graph=graph,
                                         wl_iterations=self.wl_iterations,)

        node_features = hasher.get_node_features()
        documents = self._create_documents(walks, node_features)

        model = Doc2Vec(documents,
                        vector_size=self.dimensions,
                        window=self.window_size,
                        min_count=self.min_count,
                        hs=0,
                        dm=0,
                        workers=self.workers,
                        sample=self.down_sampling,
                        epochs=self.epochs,
                        alpha=self.learning_rate,
                        negative=1,
                        seed=self.seed or 42)
        
        self._embedding = model.docvecs.vectors_docs

    def get_embedding(self, normalize=True) -> np.array:
        """Getting the node embedding."""
        embedding = self._embedding
        if normalize:
            embedding = preprocessing.normalize(embedding)            
        return embedding

