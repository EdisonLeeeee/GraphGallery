from graphgallery.functional import device
import tensorflow as tf
import torch


def test_device():
    # how about other backend?

    # tf
    assert isinstance(device("cpu", "tf"), str)
    assert device() == 'cpu'
    assert device("cpu", "tf") == 'CPU'
    assert device("cpu", "tf") == 'cpu'
    assert device("device/cpu", "tf") == 'cpu'

    try:
        assert device("gpu", "tf") == 'GPU'
        assert device("cuda", "tf") == 'GPU'
    except RuntimeError:
        pass
    device = tf.device("cpu")
    assert device(device, "tf") == device._device_name

    # ?? torch
    device = device("cpu", "torch")
    assert isinstance(device, torch.device) and 'cpu' in str(device)
    device = device(backend="torch")
    assert isinstance(device, torch.device) and 'cpu' in str(device)

    try:
        assert 'cuda' in str(device("gpu", "torch"))
        assert 'cuda' in str(device("cuda", "torch"))
    except RuntimeError:
        pass
    device = torch.device("cpu")
    assert device(device, "torch") == device
    
    
if __name__ == "__main__":
    test_device()    
