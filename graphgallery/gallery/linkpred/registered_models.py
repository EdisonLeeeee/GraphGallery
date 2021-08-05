from graphgallery import functional as gf

TensorFlow = gf.Registry("TensorFlow-Gallery (Link Prediction)")
PyTorch = gf.Registry("PyTorch-Gallery (Link Prediction)")
PyG = gf.Registry("PyG-Gallery (Link Prediction)")
DGL = gf.Registry("DGL-PyTorch-Gallery (Link Prediction)")

MAPPING = dict(tensorflow=TensorFlow,
               pytorch=PyTorch,
               pyg=PyG,
               dgl=DGL)
