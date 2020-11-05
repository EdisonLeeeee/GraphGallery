from graphgallery.functional import parse_device
import tensorflow as tf
import torch

def test_device():
    # how about other backend?

    # tf
    assert isinstance(parse_device("cpu", "tf"), str)
    assert parse_device() == 'CPU:0'
    assert parse_device("cpu", "tf") == 'CPU'
    assert parse_device("cpu:0", "tf") == 'CPU:0'
    assert parse_device("device/cpu:0", "tf") == 'CPU:0'

    try:
        assert parse_device("gpu", "tf") == 'GPU'
        assert parse_device("cuda", "tf") == 'GPU'
    except RuntimeError:
        pass
    device = tf.device("cpu:0")
    assert parse_device(device, "tf") == device._device_name

    #?? torch
    device = parse_device("cpu", "torch")
    assert isinstance(device, torch.device) and 'cpu' in str(device)
    device = parse_device(backend="torch")
    assert isinstance(device, torch.device) and 'cpu' in str(device)

    try:
        assert 'cuda' in str(parse_device("gpu", "torch"))
        assert 'cuda' in str(parse_device("cuda", "torch"))
    except RuntimeError:
        pass
    device = torch.device("cpu")
    assert parse_device(device, "torch") == device






