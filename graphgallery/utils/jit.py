"""This is modified from torch_geometric, but haven't been used"""

import numba


def jit(**kwargs):
    def decorator(func):
        try:
            return numba.jit(cache=True, **kwargs)(func)
        except RuntimeError:
            return numba.jit(cache=False, **kwargs)(func)

    return decorator
