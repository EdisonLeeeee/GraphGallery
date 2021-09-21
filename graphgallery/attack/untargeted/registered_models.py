from graphgallery import functional as gf

TensorFlow = gf.Registry("TensorFlow-Attacker")
PyTorch = gf.Registry("PyTorch-Attacker")
Common = gf.Registry("Common-Attacker")

MAPPING = {"tensorflow": TensorFlow,
           "pytorch": PyTorch}
