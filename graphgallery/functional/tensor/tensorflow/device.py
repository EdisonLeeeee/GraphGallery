import tensorflow as tf

__all__ = ['cpu', 'gpu']


def cpu(cpu_id: int = 0) -> str:
    return f'CPU:{cpu_id}'


def gpu(gpu_id: int = 0) -> str:
    return f'GPU:{gpu_id}'
