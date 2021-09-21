import torch

__all__ = ['cpu', 'gpu']


def cpu(cpu_id: int = 0) -> str:
    return torch.device(f'cpu:{cpu_id}')


def gpu(gpu_id: int = 0) -> str:
    return torch.device(f'cuda:{gpu_id}')
