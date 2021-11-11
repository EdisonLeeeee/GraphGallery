import torch

__all__ = ['empty_cache', 'max_memory', 'gpu_memory']


def empty_cache():
    torch.cuda.empty_cache()


def max_memory():
    """return the maximum allocated memory for all variables.

    Returns
    -------
    allocate memory in bytes

    Note
    ----
    Not for Windows users.
    """
    import resource
    memory = 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return memory


def gpu_memory():
    """return the maximum allocated GPU memory

    Returns
    -------
    allocated GPU memory in bytes.
    """
    memory = torch.cuda.max_memory_allocated()
    return memory
