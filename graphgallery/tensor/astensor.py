from graphgallery.tensor.tf_tensor import astftensor, astftensors


def astensor(x, dtype=None):
    """Convert input matrices to Tensor or SparseTensor.
    """
    #TODO: torch
    return astftensor(x, dtype=dtype)


def astensors(*xs):
    """Convert input matrices to Tensor(s) or SparseTensor(s).

    Parameters:
    ----------
    xs: tf.Tensor, tf.Variable, Scipy sparse matrix, 
        Numpy array-like, or a list of them, etc.

    Returns:
    ----------      
        Tensor(s) or SparseTensor(s) with dtype:       
        1. `graphgallery.floatx()` if `x` in `xs` is floating
        2. `graphgallery.intx() ` if `x` in `xs` is integer
        3. `Bool` if `x` in `xs` is bool.
    """
    #TODO: torch
    return astftensors(*xs)
