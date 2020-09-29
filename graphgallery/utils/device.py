import os.path as osp
import torch

def parse_device(device: str, kind: str) -> str:
    # TODO:
    # check if gpu is available
    
    # tf.device and torch.device
    if hasattr(device, '_device_name') or hasattr(device, 'type'):
        return device
    if device is None:
        if kind == "T":
            return 'CPU'
        else:
            return torch.device('cpu')
        
    
    _device = osp.split(device.lower())[1]
    if not any((_device.startswith("cpu"),
                _device.startswith("cuda"),
                _device.startswith("gpu"))):
        raise RuntimeError(
            f" Expected one of cpu (CPU), cuda (CUDA), gpu (GPU) device type at start of device string: {device}")

    if _device.startswith("cuda"):
        if kind == "T":
            _device = "GPU" + _device[4:]
    elif _device.startswith("gpu"):
        if kind == "P":
            _device = "cuda" + _device[3:]

    if kind == "P":
        if _device.startswith('cuda'):
            torch.cuda.empty_cache()
        return torch.device(_device)
    return _device
