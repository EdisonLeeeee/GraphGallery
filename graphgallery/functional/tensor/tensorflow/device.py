import tensorflow as tf

__all__ = ['cpu', 'gpu']


def cpu(cpu_id: int = 0) -> str:
    """
    Returns the cpu number.

    Args:
        cpu_id: (str): write your description
    """
    return f'CPU:{cpu_id}'


def gpu(gpu_id: int = 0) -> str:
    """
    Return a random integer.

    Args:
        gpu_id: (str): write your description
    """
    return f'GPU:{gpu_id}'
