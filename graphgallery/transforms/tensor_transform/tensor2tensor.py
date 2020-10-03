import torch
from graphgallery import transforms as T
from graphgallery.utils.device import parse_device
from graphgallery import is_tf_tensor, is_th_tensor

def tensor2tensor(tensor, device=None):
    """Convert a TensorFLow tensor to PyTorch Tensor,
    or vice versa
    """
    if is_tf_tensor(tensor):
        m = T.tensoras(tensor)
        device = parse_device(device, "P")
        return T.th_tensor.astensor(m, device=device)
    elif is_th_tensor(tensor):
        m = T.tensoras(tensor)
        device = parse_device(device, "T")
        return T.tf_tensor.astensor(m, device=device)
    else:
        raise ValueError(f"The input must be a Tensorflow Tensor or PyTorch Tensor, buf got {type(tensor)}")

