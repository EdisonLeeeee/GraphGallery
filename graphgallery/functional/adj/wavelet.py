import scipy.sparse as sp
import numpy as np

from sklearn.preprocessing import normalize

from ..transforms import Transform
from ..decorators import MultiInputs

# Version: Compute the exact signature
# def laplacian(W, normalized=True):
#     """Return the Laplacian of the weight matrix."""
#     # Degree matrix.
#     d = W.sum(axis=0).A1
#     # Laplacian matrix.
#     if not normalized:
#         D = sp.diags(d)
#         L = D - W
#     else:
#         d = 1 / np.sqrt(d)
#         D = sp.diags(d)
#         I = sp.identity(d.size, dtype=W.dtype)
#         L = I - D * W * D

#     return L

# def fourier(L, algo='eigh', k=100):
#     """Return the Fourier basis, i.e. the EVD of the Laplacian."""
#     def sort(lamb, U):
#         idx = lamb.argsort()
#         return lamb[idx], U[:, idx]

#     if algo is 'eig':
#         lamb, U = np.linalg.eig(L.toarray())
#     elif algo is 'eigh':
#         lamb, U = np.linalg.eigh(L.toarray())
#     elif algo is 'eigs':
#         lamb, U = sp.linalg.eigs(L, k=k, which='SM')
#     elif algo is 'eigsh':
#         lamb, U = sp.linalg.eigsh(L, k=k, which='SM')
#     else:
#         raise ValueError(f'Invalid argument of {algo}.')

#     lamb, U = sort(lamb, U)

#     return lamb, U

# def weight_wavelet(wavelet_s, lamb, U):
#     # In original code, lamb has changed but here hasn't
#     lamb = np.exp(-lamb*wavelet_s)
#     Weight = (U * lamb) @ U.T
#     return Weight
# #     return Weight, lamb

# def weight_wavelet_inverse(wavelet_s, lamb, U):
#     lamb = np.exp(lamb*wavelet_s)
#     Weight = (U * lamb) @ U.T
#     return Weight


# def wavelet_basis(adj_matrix, wavelet_s=1.0, laplacian_normalize=True,
#                   sparseness=True, threshold=1e-4, weight_normalize=False, k=100):

#     L = laplacian(adj_matrix, normalized=laplacian_normalize)
#     lamb, U = fourier(L, k=k)
# #     Weight, lamb = weight_wavelet(wavelet_s, lamb, U)
#     Weight = weight_wavelet(wavelet_s, lamb, U)
#     inverse_Weight = weight_wavelet_inverse(wavelet_s, lamb, U)
#     del U, lamb

#     if sparseness:
#         Weight[Weight < threshold] = 0.0
#         inverse_Weight[inverse_Weight < threshold] = 0.0

#     if weight_normalize:
#         Weight = normalize(Weight, norm='l1', axis=1)
#         inverse_Weight = normalize(inverse_Weight, norm='l1', axis=1)

#     Weight = sp.csr_matrix(Weight)
#     inverse_Weight = sp.csr_matrix(inverse_Weight)
#     t_k = Weight, inverse_Weight
#     return t_k

# ==================================================================
# Version: Approximate with Chebychev polynomial

class WaveletBasis(Transform):
    def __init__(self, order=3, wavelet_s=1.2, threshold=1e-4, wavelet_normalize=True):
        """
        Initialize wavelet wavelet.

        Args:
            self: (todo): write your description
            order: (int): write your description
            wavelet_s: (str): write your description
            threshold: (float): write your description
            wavelet_normalize: (bool): write your description
        """
        super().__init__()
        self.order = order
        self.wavelet_s = wavelet_s
        self.threshold = threshold
        self.wavelet_normalize = wavelet_normalize

    def __call__(self, adj_matrix):
        """
        Return the wavelet wavelet wavelet.

        Args:
            self: (todo): write your description
            adj_matrix: (array): write your description
        """
        return wavelet_basis(adj_matrix, order=self.order, wavelet_s=self.wavelet_s,
                             threshold=self.threshold, wavelet_normalize=self.wavelet_normalize)

    def __repr__(self):
        """
        Return a repr representation of a repr__.

        Args:
            self: (todo): write your description
        """
        return f"{self.__class__.__name__}(order={self.order}, wavelet_s={self.wavelet_s}, threshold={self.threshold}, wavelet_normalize={self.wavelet_normalize})"


def laplacian(adj_matrix, normalized=True):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    d = adj_matrix.sum(axis=0).A1
    # Laplacian matrix.
    if not normalized:
        D = sp.diags(d)
        L = D - adj_matrix
    else:
        d = 1 / (np.sqrt(d) + 1e-6)
#         d[np.isinf(d)] = 0.
        D = sp.diags(d)
        I = sp.identity(d.size, dtype=adj_matrix.dtype)
        L = I - D * adj_matrix * D
    return L


def compute_cheb_coeff_basis(scale, order):
    """
    Compute the coefficients coefficients.

    Args:
        scale: (float): write your description
        order: (int): write your description
    """
    xx = np.array([np.cos((2. * i - 1.) / (2. * order) * np.pi)
                   for i in range(1, order + 1)])
    basis = [np.ones((1, order)), xx]
    for k in range(order + 1 - 2):
        basis.append(2 * np.multiply(xx, basis[-1]) - basis[-2])
    basis = np.vstack(basis)
    f = np.exp(-scale * (xx + 1))
    products = np.einsum("j,ij->ij", f, basis)
    coeffs = 2. / order * products.sum(1)
    coeffs[0] /= 2.
    return coeffs


@MultiInputs()
def wavelet_basis(adj_matrix, order=3, wavelet_s=1.0, threshold=1e-4, wavelet_normalize=False):
    """
    R compute wavelet basis.

    Args:
        adj_matrix: (todo): write your description
        order: (int): write your description
        wavelet_s: (str): write your description
        threshold: (float): write your description
        wavelet_normalize: (bool): write your description
    """
    lap = laplacian(adj_matrix)
    N = adj_matrix.shape[0]
    I = sp.eye(N)
    L = lap - I
    monome = {0: I, 1: L}

    for k in range(2, order + 1):
        monome[k] = 2 * L @ monome[k - 1] - monome[k - 2]

    def compute_walelet(tau):
        """
        Computes the coefficient matrix.

        Args:
            tau: (todo): write your description
        """
        coeffs = compute_cheb_coeff_basis(tau, order)
        w = np.sum([coeffs[k] * monome[k] for k in range(order + 1)])
        w.data = thres(w.data)
        w.eliminate_zeros()
        return w

    thres = np.vectorize(lambda x: x if x > threshold else 0.)
    Wavelet = compute_walelet(wavelet_s)
    Wavelet_inverse = compute_walelet(-wavelet_s)

    if wavelet_normalize:
        Wavelet = normalize(Wavelet, norm='l1', axis=1)
        Wavelet_inverse = normalize(Wavelet_inverse, norm='l1', axis=1)

    return Wavelet, Wavelet_inverse
