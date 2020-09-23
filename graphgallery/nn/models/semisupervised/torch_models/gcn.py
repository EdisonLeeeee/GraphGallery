import torch
import torch.nn.functional as F

from torch.nn import Module, ModuleList
from torch import optim

from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.nn.models import TorchKerasModel
from graphgallery.nn.layers import GraphConvolution
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.utils.decorators import EqualVarLength
from graphgallery import transformers as T

# a map for activation functions 
ACT2FN = {
    "relu": F.relu,
    "tanh": F.tanh,
    "elu": F.elu,
}

def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(
            "function {} not found in ACT2FN mapping {} or torch.nn.functional".format(
                activation_string, list(ACT2FN.keys())
            )
        )

class _Model(Module):
    
    def __init__(self, input_channels, hiddens, output_channels, activations=['relu'], dropouts=[0.5], lr=0.01, use_bias=False):
        super().__init__()

        # save for later usage
        self.activations = activations
        self.dropouts = dropouts

        inc, outc = input_channels, hiddens[0] 
        self.gcs = ModuleList()

        # use ModuleList to create layers with different size
        iterlist = [input_channels] + hiddens + [output_channels]
        for i in range(len(iterlist) - 1):
            inc, outc = iterlist[i], iterlist[i+1]
            self.gcs.append(GraphConvolution(inc, outc, use_bias=use_bias))

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, inputs):
        x, adj, idx = inputs        

        for i in range(len(self.gcs) - 1):
                func = get_activation(self.activations[i])
                x = func(self.gcs[i]([x, adj]))
                x = F.dropout(x, self.dropouts[i], training=self.training)
        x = self.gcs[-1]([x,adj]) # last layer

        return x[idx]
    
    def reset_parameters(self):
        for i, l in enumerate(self.gcs):
            self.gcs[i].reset_parameters()
                       
                       

class GCN(SemiSupervisedModel):
    """
        Implementation of Graph Convolutional Networks (GCN). 
        `Semi-Supervised Classification with Graph Convolutional Networks 
        <https://arxiv.org/abs/1609.02907>`
        Tensorflow 1.x implementation: <https://github.com/tkipf/gcn>
        Pytorch implementation: <https://github.com/tkipf/pygcn>

    """

    def __init__(self, *graph, adj_transformer="normalize_adj", attr_transformer=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        """Creat a Graph Convolutional Networks (GCN) model.


        This can be instantiated in several ways:

            model = GCN(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = GCN(adj_matrix, attr_matrix, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `attr_matrix` is a 2D Numpy array-like matrix denoting the node 
                 attributes, `labels` is a 1D Numpy array denoting the node labels.


        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
            A sparse, attributed, labeled graph.
        adj_transformer: string, `transformer`, or None. optional
            How to transform the adjacency matrix. See `graphgallery.transformers`
            (default: :obj:`'normalize_adj'` with normalize rate `-0.5`.
            i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
        attr_transformer: string, transformer, or None. optional
            How to transform the node attribute matrix. See `graphgallery.transformers`
            (default :obj: `None`)
        device: string. optional 
            The device where the model is running on. You can specified `CPU` or `GPU` 
            for the model. (default: :str: `CPU:0`, i.e., running on the 0-th `CPU`)
        seed: interger scalar. optional 
            Used in combination with `tf.random.set_seed` & `np.random.seed` 
            & `random.seed` to create a reproducible sequence of tensors across 
            multiple calls. (default :obj: `None`, i.e., using random seed)
        name: string. optional
            Specified name for the model. (default: :str: `class.__name__`)
        kwargs: other customed keyword Parameters.
        """

        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        self.adj_transformer = T.get(adj_transformer)
        self.attr_transformer = T.get(attr_transformer)
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transformer(graph.adj_matrix)
        attr_matrix = self.attr_transformer(graph.attr_matrix)

        self.feature_inputs, self.structure_inputs = T.astensors(
            attr_matrix, adj_matrix)

    # use decorator to make sure all list arguments have the same length
    @EqualVarLength()
    def build(self, hiddens=[16], activations=['relu'], dropouts=[0.5],
              l2_norms=[5e-4], lr=0.01, use_bias=False):
        
        self.model = _Model(self.graph.n_attrs, hiddens, self.graph.n_classes, activations=activations, dropouts=dropouts, lr=lr, use_bias=use_bias).to(self.device)
        
        
    def train_sequence(self, index):
        index = T.asintarr(index)
        labels = self.graph.labels[index]
        sequence = FullBatchNodeSequence(
            [self.feature_inputs, self.structure_inputs, index], labels, device=self.device)

        return sequence
