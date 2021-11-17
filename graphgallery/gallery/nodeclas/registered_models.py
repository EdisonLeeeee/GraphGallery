from graphgallery import functional as gf

PyTorch = gf.Registry("PyTorch-Gallery (Node Classification)")
PyG = gf.Registry("PyG-Gallery (Node Classification)")
DGL = gf.Registry("DGL-PyTorch-Gallery (Node Classification)")

MAPPING = dict(pytorch=PyTorch,
               pyg=PyG,
               dgl=DGL)
