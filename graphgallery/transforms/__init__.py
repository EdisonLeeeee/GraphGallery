# from graphgallery.transforms import adj_transform
# from graphgallery.transforms import attr_transform
# from graphgallery.transforms import graph_transform

# edge_transform
from graphgallery.transforms.edge_transform.edge_transpose import edge_transpose
from graphgallery.transforms.edge_transform.add_selfloops_edge import add_selfloops_edge
from graphgallery.transforms.edge_transform.augment_edge import augment_edge
from graphgallery.transforms.edge_transform.normalize_edge import normalize_edge

# adj_transform
from graphgallery.transforms.base_transform import Transform, NullTransformer
from graphgallery.transforms.graph_partition import GraphPartition, graph_partition
from graphgallery.transforms.adj_transform.normalize_adj import NormalizeAdj, normalize_adj
from graphgallery.transforms.adj_transform.add_selfloops import AddSelfLoops, add_selfloops
from graphgallery.transforms.adj_transform.wavelet import WaveletBasis, wavelet_basis
from graphgallery.transforms.adj_transform.chebyshef import ChebyBasis, cheby_basis
from graphgallery.transforms.adj_transform.neighbor_sampler import NeighborSampler, neighbor_sampler
from graphgallery.transforms.adj_transform.gdc import GDC, gdc
from graphgallery.transforms.adj_transform.svd import SVD, svd
from graphgallery.transforms.adj_transform.adj2edges import sparse_adj_to_sparse_edges, SparseAdjToSparseEdges
from graphgallery.transforms.adj_transform.augment_adj import augment_adj
from graphgallery.transforms.adj_transform.sparse_reshape import SparseReshape, sparse_reshape


# attr_transform
from graphgallery.transforms.attr_transform.normalize_attr import NormalizeAttr, normalize_attr
from graphgallery.transforms.attr_transform.augment_attr import augment_attr



from graphgallery.transforms.edge_transform.edges2adj import (sparse_edges_to_sparse_adj,
                                                                  SparseEdgesToSparseAdj)

# tensor transform
from graphgallery.transforms.tensor_transform.astensor import (astensor, astensors,
                                                                   sparse_tensor_to_sparse_adj,
                                                                   sparse_edges_to_sparse_tensor,
                                                                   normalize_adj_tensor,
                                                                   normalize_edge_tensor)

# other transform
from graphgallery.transforms.get_transform import get, Compose
from graphgallery.transforms.transform import indices2mask, asintarr
