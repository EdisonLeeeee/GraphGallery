from graphgallery import functional as gf

TensorFlow = gf.Registry("TensorFlow-Gallery (Node Classification)")
PyTorch = gf.Registry("PyTorch-Gallery (Node Classification)")
PyG = gf.Registry("PyG-Gallery (Node Classification)")
DGL = gf.Registry("DGL-PyTorch-Gallery (Node Classification)")
Common = gf.Registry("Common-Gallery (Node Classification)")

MAPPING = dict(tensorflow=TensorFlow,
               pytorch=PyTorch,
               pyg=PyG,
               dgl=DGL)
