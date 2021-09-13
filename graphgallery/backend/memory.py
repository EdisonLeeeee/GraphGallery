import torch

__all__ = ['set_memory_growth', 'empty_cache', 'max_memory', 'gpu_memory']


def set_memory_growth():
    """Set if memory growth should be enabled for ALL `PhysicalDevice`.

    If memory growth is enabled for ALL `PhysicalDevice`, the runtime initialization
    will not allocate all memory on the device. Memory growth cannot be configured
    on a `PhysicalDevice` with virtual devices configured.

    """
    backend = gg.backend()
    if backend != 'tensorflow':
        return

    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def empty_cache():
    torch.cuda.empty_cache()
    # tf.keras.backend.clear_session()


def max_memory():
    """return the maximum allocated memory for all variables

    Returns
    -------
    allocate memory in bytes
    """
    import resource
    memory = 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return memory


def gpu_memory():
    """return the maximum allocated GPU memory for PyTorch backend,
    but it currently not works for TensorFlow backend.

    Returns
    -------
    allocated GPU memory in bytes.
    """
    memory = torch.cuda.max_memory_allocated()
    return memory
