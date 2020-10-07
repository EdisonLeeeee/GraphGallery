import torch
import tensorflow as tf
from graphgallery import backend
from graphgallery.utils.raise_error import assert_kind


def parse_device(device: str = None, kind: str = None) -> str:
    """
    Specifies the device for corresponding kind 

    Parameters
    ----------
    device : (string, tf.device, torch.device, None) 
        device name such as 'cpu', 'gpu', 'cuda'
        or an instance of tf.device/torch.device
    kind : string 
        kind of backend for device
        'T' for tensorflow, 'P' for pytorch

    Returns
    -------
    return string for tf backend
    return torch.device instance for torch backend
    """

    if kind is None:
        kind = backend().kind
    else:
        assert_kind(kind)
    if device is None:
        # by default, return CPU device
        if kind == "T":
            return 'CPU:0'
        else:
            return torch.device('cpu')

    # existing tensorflow device
    if (hasattr(device, '_device_name') and kind == "T"):
        return device._device_name
    # existing pytorch device
    if (isinstance(device, torch.device) and kind == "P"):
        return device

    if hasattr(device, '_device_name'):
        # tensorflow device meets pytorch backend
        _device = device._device_name.split('/')[-1]
    elif isinstance(device, torch.device):
        # pytorch device meets tensorflow backend
        _device = str(device)
    else:
        _device = str(device).lower().split('/')[-1]
        if not any(
            (_device.startswith("cpu"), _device.startswith("cuda"), _device.startswith("gpu"))):
            raise RuntimeError(
                f" Expected one of cpu (CPU), cuda (CUDA), gpu (GPU) at the start of device string, bot got {device}."
            )

    # modify _device name
    if _device.startswith("cuda") and kind == "T":
        _device = "GPU" + _device[4:]  # tensorflow uses 'GPU' instead of 'cuda'
    elif _device.startswith("gpu") and kind == "P":
        _device = "cuda" + _device[3:]  # pytorch uses 'cuda' instead of 'GPU'

    # pytorch return torch.device
    if kind == "P":
        if _device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA is unavailable for PyTorch backend.")
            # empty cache to avoid unnecessary memory usage
            # TODO: is this necessary?
            torch.cuda.empty_cache()
        return torch.device(_device)

    # tf return string
    if _device.startswith('gpu') and not tf.config.list_physical_devices('GPU'):
        raise RuntimeError(f"GPU is unavailable for TensorFlow backend.")
    return _device.upper()
