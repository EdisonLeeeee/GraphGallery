from graphgallery.utils.device import parse_device
import tensorflow as tf
import torch

def test_device():
    # tf
    assert isinstance(parse_device("cpu", "T"), str)
    assert parse_device() == 'CPU:0'
    assert parse_device("cpu", "T") == 'CPU'
    assert parse_device("cpu:0", "T") == 'CPU:0'
    assert parse_device("device/cpu:0", "T") == 'CPU:0'
    #?? without gpu
    try:
        assert parse_device("gpu", "T") == 'GPU'
        assert parse_device("cuda", "T") == 'GPU'
    except RuntimeError:
        pass
    device = tf.device("cpu:0")
    assert parse_device(device, "T") == device._device_name

    #?? torch
    device = parse_device("cpu", "P")
    assert isinstance(device, torch.device) and str(device) == 'cpu'
    device = parse_device(kind="P")
    assert isinstance(device, torch.device) and str(device) == 'cpu'
    #?? without gpu
    try:
        assert str(parse_device("gpu", "P")) == 'cuda'
        assert str(parse_device("cuda", "P")) == 'cuda'
    except RuntimeError:
        pass
    device = torch.device("cpu")
    assert parse_device(device, "P") == device






