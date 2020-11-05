import torch

__all__ = ['cpu', 'gpu']


def cpu(cpu_id: int = 0) -> str:
    """
    Get the cpu.

    Args:
        cpu_id: (str): write your description
    """
    return torch.device(f'cpu:{cpu_id}')


def gpu(gpu_id: int = 0) -> str:
    """
    Get a random gpu device.

    Args:
        gpu_id: (str): write your description
    """
    return torch.device(f'cuda:{gpu_id}')
