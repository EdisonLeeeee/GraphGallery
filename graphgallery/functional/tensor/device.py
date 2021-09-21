import torch
import graphgallery as gg

from .pytorch import device as th_device

__all__ = ['device']


def device(device=None, backend=None):
    """
    Specify the device for the corresponding backend

    Parameters
    ----------
    device: (string, tf.device, torch.device, None)
        device name such as 'cpu', 'gpu', 'cuda'
        or an instance of 'tf.device'/'torch.device'
    backend: String or BackendModule, optional.
        'tensorflow', 'torch', TensorFlowBackend,
        PyTorchBackend, etc. if not specified,
        return the current backend module.

    Returns
    -------
    device:
    - 'string' for tensorflow backend,
    - 'torch.device' instance for pytorch backend
    """
    backend = gg.backend(backend)

    if device is None:
        # by default, return CPU device
        if backend == "tensorflow":
            from .tensorflow import device as tf_device
            return tf_device.cpu()
        elif backend == "torch":
            return th_device.cpu()
        else:
            raise RuntimeError("This may not happen!")

    # existing tensorflow device
    if hasattr(device, '_device_name'):
        _device = device._device_name
    # existing pytorch device
    elif isinstance(device, torch.device):
        _device = str(device)

    _device = str(device).lower().split('/')[-1]
    _device, *_device_id = _device.split(":")
    _device_id = "".join(_device_id)

    if not _device in {"cpu", "cuda", "gpu"}:
        raise RuntimeError(
            f"Expected one of cpu (CPU), cuda (CUDA), gpu (GPU) at the start of device string, but got {device}."
        )
    if not _device_id:
        _device_id = 0
    else:
        try:
            _device_id = int(_device_id)
        except ValueError as e:
            raise ValueError(f"Invalid device id in {device}.")

    # pytorch backend returns 'torch.device'
    if backend == "torch":
        if _device == "cpu":
            return th_device.cpu(_device_id)
        else:
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA is unavailable for {backend}.")
            # empty cache to avoid unnecessary memory usage
            # TODO: is this necessary?
            torch.cuda.empty_cache()
            return th_device.gpu(_device_id)

    # tensorflow backend returns 'string'
    from .tensorflow import device as tf_device
    tf = backend.module
    if _device == "cpu":
        return tf_device.cpu(_device_id)
    # FIXME: Tensorflow 2.4.0 requires cuDNN 8.0 and CUDA 11.0
    # while Tensorflow  2.1~2.3 require cuDNN 7.6 and CUDA 10.1
    elif not tf.config.list_physical_devices('GPU'):
        raise RuntimeError(f"GPU is unavailable for {backend}.")
    return tf_device.gpu(_device_id)
