from graphgallery import backend

_BACKEND = backend()

# Tensorflow layers
if _BACKEND .kind == "T":

    from graphgallery.nn.layers.TF.gcn import GraphConvolution
    from graphgallery.nn.layers.TF.sgc import SGConvolution
    from graphgallery.nn.layers.TF.gat import GraphAttention
    from graphgallery.nn.layers.TF.gwnn import WaveletConvolution
    from graphgallery.nn.layers.TF.robustgcn import GaussionConvolution_F, GaussionConvolution_D
    from graphgallery.nn.layers.TF.graphsage import MeanAggregator, GCNAggregator
    from graphgallery.nn.layers.TF.chebynet import ChebyConvolution
    from graphgallery.nn.layers.TF.densegcn import DenseConvolution
    from graphgallery.nn.layers.TF.top_k import Top_k_features
    from graphgallery.nn.layers.TF.lgcn import LGConvolution
    from graphgallery.nn.layers.TF.edgeconv import GraphEdgeConvolution
    from graphgallery.nn.layers.TF.mediansage import MedianAggregator, MedianGCNAggregator
    from graphgallery.nn.layers.TF.gcna import GraphConvAttribute
    from graphgallery.nn.layers.TF.dagnn import PropConvolution
    from graphgallery.nn.layers.TF.misc import SparseConversion, Scale, Sample, Gather, Laplacian, Mask

else:
    from graphgallery.nn.layers.PTH.gcn import GraphConvolution
