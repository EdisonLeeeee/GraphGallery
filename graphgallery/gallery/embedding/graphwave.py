import pygsp
import numpy as np

class GraphWave:
    r"""An implementation of `"GraphWave" <https://dl.acm.org/citation.cfm?id=2806512>`_
    from the KDD '18 paper "Learning Structural Node Embeddings Via Diffusion Wavelets".
    The procedure first calculates the graph wavelets using a heat kernel. The wavelets
    are treated as probability distributions over nodes from a source node. Using these
    the characteristic function is evaluated at certain gird points to learn structural
    node embeddings of the vertices.
    """

    def __init__(self, sample_number: int = 200, step_size: float = 0.1, heat_coefficient: float = 1.0,
                 approximation: int = 100, mechanism: str = "approximate", switch: int = 1000, seed: int = None):

        self.sample_number = sample_number
        self.step_size = step_size
        self.heat_coefficient = heat_coefficient
        self.approximation = approximation
        self.mechanism = mechanism
        self.switch = switch
        self.seed = seed

    def _create_evaluation_points(self):
        """
        Calculating the grid points.
        """
        self._steps = [x * self.step_size for x in range(self.sample_number)]

    def _check_size(self, graph):
        """
        Checking the size of the target graph. Switching based on size and settings.
        """
        self._number_of_nodes = graph.shape[0]
        if self._number_of_nodes > self.switch:
            self.mechanism = "approximate"

    def _single_wavelet_generator(self, node):
        """
        Calculating the characteristic function for a given node, using the eigen decomposition.
        """
        impulse = np.zeros((self._number_of_nodes))
        impulse[node] = 1.0
        diags = np.diag(np.exp(-self.heat_coefficient * self._eigen_values))
        eigen_diag = np.dot(self._eigen_vectors, diags)
        waves = np.dot(eigen_diag, np.transpose(self._eigen_vectors))
        wavelet_coefficients = np.dot(waves, impulse)
        return wavelet_coefficients

    def _exact_wavelet_calculator(self):
        """
        Calculates the structural role embedding using the exact eigenvalue decomposition.
        """
        self._real_and_imaginary = []
        for node in range(self._number_of_nodes):
            wave = self._single_wavelet_generator(node)
            wavelet_coefficients = [np.mean(np.exp(wave * 1.0 * step * 1j)) for step in self._steps]
            self._real_and_imaginary.append(wavelet_coefficients)
        self._real_and_imaginary = np.array(self._real_and_imaginary)

    def _exact_structural_wavelet_embedding(self):
        """
        Calculates the eigenvectors, eigenvalues and an exact embedding is created.
        """
        self._G.compute_fourier_basis()
        self._eigen_values = self._G.e / max(self._G.e)
        self._eigen_vectors = self._G.U
        self._exact_wavelet_calculator()

    def _approximate_wavelet_calculator(self):
        """
        Given the Chebyshev polynomial and graph the approximate embedding is calculated.
        """
        self._real_and_imaginary = []
        for node in range(self._number_of_nodes):
            impulse = np.zeros((self._number_of_nodes))
            impulse[node] = 1
            wave_coeffs = pygsp.filters.approximations.cheby_op(self._G, self._chebyshev, impulse)
            real_imag = [np.mean(np.exp(wave_coeffs * 1 * step * 1j)) for step in self._steps]
            self._real_and_imaginary.append(real_imag)
        self._real_and_imaginary = np.array(self._real_and_imaginary)

    def _approximate_structural_wavelet_embedding(self):
        """
        Estimating the largest eigenvalue.
        Setting up the heat filter and the Cheybshev polynomial.
        Using the approximate wavelet calculator method.
        """
        self._G.estimate_lmax()
        self._heat_filter = pygsp.filters.Heat(self._G, tau=[self.heat_coefficient])
        self._chebyshev = pygsp.filters.approximations.compute_cheby_coeff(self._heat_filter,
                                                                           m=self.approximation)
        self._approximate_wavelet_calculator()

    def fit(self, graph):
        """
        Fitting a GraphWave model.
        """
        # graph.remove_edges_from(nx.selfloop_edges(graph))
        self._create_evaluation_points()
        self._check_size(graph)
        self._G = pygsp.graphs.Graph(graph)

        if self.mechanism == "exact":
            self._exact_structural_wavelet_embedding()
        elif self.mechanism == "approximate":
            self._approximate_structural_wavelet_embedding()
        else:
            raise NameError("Unknown method.")

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.
        """
        embedding = [self._real_and_imaginary.real, self._real_and_imaginary.imag]
        embedding = np.concatenate(embedding, axis=1)
        return embedding
