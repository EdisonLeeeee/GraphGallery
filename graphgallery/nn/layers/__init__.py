from graphgallery import backend

_BACKEND = backend()

# Tensorflow layers
if _BACKEND .kind == "T":
    
    from graphgallery.nn.layers.tf_layers.gcn import GraphConvolution
    from graphgallery.nn.layers.tf_layers.sgc import SGConvolution
    from graphgallery.nn.layers.tf_layers.gat import GraphAttention
    from graphgallery.nn.layers.tf_layers.gwnn import WaveletConvolution
    from graphgallery.nn.layers.tf_layers.robustgcn import GaussionConvolution_F, GaussionConvolution_D
    from graphgallery.nn.layers.tf_layers.graphsage import MeanAggregator, GCNAggregator
    from graphgallery.nn.layers.tf_layers.chebynet import ChebyConvolution
    from graphgallery.nn.layers.tf_layers.densegcn import DenseConvolution
    from graphgallery.nn.layers.tf_layers.top_k import Top_k_features
    from graphgallery.nn.layers.tf_layers.lgcn import LGConvolution
    from graphgallery.nn.layers.tf_layers.edgeconv import GraphEdgeConvolution
    from graphgallery.nn.layers.tf_layers.mediansage import MedianAggregator, MedianGCNAggregator
    from graphgallery.nn.layers.tf_layers.gcnf import GraphConvattribute
    from graphgallery.nn.layers.tf_layers.dagnn import PropConvolution
    from graphgallery.nn.layers.tf_layers.misc import SparseConversion, Scale, Sample, Gather, Laplacian
    
else:
    ...
