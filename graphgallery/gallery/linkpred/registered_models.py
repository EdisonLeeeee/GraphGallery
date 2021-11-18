from graphgallery import functional as gf

PyTorch = gf.Registry("PyTorch-Gallery (Link Prediction)")
PyG = gf.Registry("PyG-Gallery (Link Prediction)")
DGL = gf.Registry("DGL-PyTorch-Gallery (Link Prediction)")

MAPPING = dict(pytorch=PyTorch,
               pyg=PyG,
               dgl=DGL)
