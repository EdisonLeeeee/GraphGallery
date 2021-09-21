import numpy as np
import scipy.sparse as sp
from collections import Counter
from sklearn.preprocessing import normalize


class NodeSketch:
    r"""An implementation of `"NodeSketch" <https://exascale.info/assets/pdf/yang2019nodesketch.pdf>`_
    from the KDD '19 paper "NodeSketch: Highly-Efficient Graph Embeddings
    via Recursive Sketching". The procedure  starts by sketching the self-loop-augmented 
    adjacency matrix of the graph to output low-order node embeddings, and then recursively 
    generates k-order node embeddings based on the self-loop-augmented adjacency matrix 
    and (k-1)-order node embeddings.
    """

    def __init__(self, dimensions: int = 32, iterations: int = 2, decay: float = 0.01, seed: int = None):

        self.dimensions = dimensions
        self.iterations = iterations
        self.decay = decay
        self.seed = seed
        self._weight = self.decay / self.dimensions

    def _generate_hash_values(self):
        """
        Predefine a hash matrix
        """
        random_matrix = np.random.rand(self.dimensions, self._num_nodes)
        hashes = -np.log(random_matrix)
        return hashes

    def _do_single_sketch(self):
        """
        Perform a single round of sketching
        """
        sketch = []
        for iter in range(self.dimensions):
            hashed = self._sla.copy()
            hashed.data = np.array([self._hash_values[iter, self._sla.col[edge]] / self._sla.data[edge] for edge in range(len(self._sla.data))])
            min_values = [np.inf for k in range(self._num_nodes)]
            min_indices = [None for k in range(self._num_nodes)]
            for i, j, v in zip(hashed.row, hashed.col, hashed.data):
                if v < min_values[i]:
                    min_values[i] = v
                    min_indices[i] = j
            sketch.append(min_indices)
        self._sketch = sketch

    def _augment_sla(self):
        """
        Augment the sla matrix based on the previous sketch
        """
        self._sla = self._sla_original.copy()
        data = []
        row = []
        col = []
        for node in range(self._num_nodes):
            frequencies = []
            for neighbor in self._graph[node].indices:
                frequencies.append(Counter([dim[neighbor] for dim in self._sketch]))
            frequencies = sum(frequencies, Counter())
            for target, value in frequencies.items():
                row.append(node)
                col.append(target)
                data.append(value * self._weight)
        self._sla.data = np.append(self._sla.data, data)
        self._sla.row = np.append(self._sla.row, row)
        self._sla.col = np.append(self._sla.col, col)
        self._sla.sum_duplicates()

    def fit(self, graph: sp.csr_matrix):
        """
        Fitting a NodeSketch model.
        """
        self._graph = graph
        self._num_nodes = graph.shape[0]
        self._hash_values = self._generate_hash_values()
        self._sla = graph.tocoo()
        self._sla.data = np.array([1 for _ in range(len(self._sla.data))])
        self._sla_original = self._sla.copy()
        self._do_single_sketch()
        for _ in range(self.iterations - 1):
            self._augment_sla()
            self._do_single_sketch()

    def get_embedding(self, normalize=True) -> np.array:
        """Getting the node embedding."""
        embedding = np.transpose(self._sketch)
        if normalize:
            embedding = preprocessing.normalize(embedding)
        return embedding
