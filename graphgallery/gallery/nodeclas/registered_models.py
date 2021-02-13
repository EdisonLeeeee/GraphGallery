from graphgallery import functional as gf

TensorFlow = gf.Registry("TensorFlow-Gallery (Node Classification)")
PyTorch = gf.Registry("PyTorch-Gallery (Node Classification)")
PyG = gf.Registry("PyG-Gallery (Node Classification)")
DGL_PyTorch = gf.Registry("DGL-PyTorch-Gallery (Node Classification)")
DGL_TensorFlow = gf.Registry("DGL-TensorFlow-Gallery (Node Classification)")
Common = gf.Registry("Common-Gallery (Node Classification)")

MAPPING = dict(tensorflow=TensorFlow,
               pytorch=PyTorch,
               pyg=PyG,
               dgl_torch=DGL_PyTorch,
               dgl_tf=DGL_TensorFlow)
