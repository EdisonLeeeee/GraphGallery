import torch
import tensorflow as tf

__all__ = ['set_memory_growth', 'empty_cache']


def set_memory_growth() -> None:
    """Set if memory growth should be enabled for ALL `PhysicalDevice`.

    If memory growth is enabled for ALL `PhysicalDevice`, the runtime initialization
    will not allocate all memory on the device. Memory growth cannot be configured
    on a `PhysicalDevice` with virtual devices configured.

    """
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


def empty_cache() -> None:
    torch.cuda.empty_cache()
    tf.keras.backend.clear_session()
