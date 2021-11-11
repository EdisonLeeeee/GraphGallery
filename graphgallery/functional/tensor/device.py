import torch

__all__ = ['device']


def device(device=None, backend=None):
    """
    Specify the device for the corresponding backend

    Parameters
    ----------
    device: (string, tf.device, torch.device, None)
        device name such as 'cpu', 'gpu', 'cuda'
        or an instance of 'torch.device'
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

    if device is None:
        # by default, return CPU device
        return torch.device('cpu')

    # existing pytorch device
    if isinstance(device, torch.device):
        return device

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

    if _device == "cpu":
        return torch.device(f'cpu:{_device_id}')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for {backend}.")
        # empty cache to avoid unnecessary memory usage
        # TODO: is this necessary?
        # torch.cuda.empty_cache()
        return torch.device(f'cuda:{_device_id}')
