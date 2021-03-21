import numpy as np
import hashlib

class WeisfeilerLehmanHashing:
    """
    Weisfeiler-Lehman feature extractor class.
    """

    def __init__(self, graph, wl_iterations: int):
        """
        Initialization method which also executes feature extraction.
        """
        self.wl_iterations = wl_iterations
        self.graph = graph
        self._set_features()
        self._do_recursions()

    def _set_features(self):
        """
        Creating the features.
        """
        degree = self.graph.sum(1).A1
        self.features = dict(zip(np.arange(degree.size), degree))
        self.extracted_features = {k: [str(v)] for k, v in enumerate(degree)}

    def _do_a_recursion(self):
        """
        The method does a single WL recursion.
        """
        new_features = {}
        for node in range(self.graph.shape[0]):
            nebs = self.graph[node].indices
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = {k: self.extracted_features[k] + [v] for k, v in new_features.items()}
        return new_features

    def _do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.wl_iterations):
            self.features = self._do_a_recursion()

    def get_node_features(self):
        """
        Return the node level features.
        """
        return self.extracted_features

    def get_graph_features(self):
        """
        Return the graph level features.
        """
        return [feature for node, features in self.extracted_features.items() for feature in features]
