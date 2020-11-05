import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import DenseConvolution, Gather
from graphgallery.nn.gallery import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery import functional as F


class SAT(SemiSupervisedModel):

    def __init__(self, *graph, adj_transform="normalize_adj", 
                 attr_transform=None, k=35,
                 device='cpu:0', seed=None, name=None, **kwargs):
        """Create a Graph Convolutional Networks (GCN) model
            using Spetral Adversarial Training (SAT) defense strategy.


        This can be instantiated in several ways:

            model = SAT(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = SAT(adj_matrix, attr_matrix, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `attr_matrix` is a 2D Numpy array-like matrix denoting the node 
                 attributes, `labels` is a 1D Numpy array denoting the node labels.


        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
            A sparse, attributed, labeled graph.
        adj_transform: string, `transform`, or None. optional
            How to transform the adjacency matrix. See `graphgallery.functional`
            (default: :obj:`'normalize_adj'` with normalize rate `-0.5`.
            i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
        attr_transform: string, `transform`, or None. optional
            How to transform the node attribute matrix. See `graphgallery.functional`
            (default :obj: `None`)
        k: integer. optional.
            The number of eigenvalues and eigenvectors desired.
            `k` must be smaller than N-1. It is not possible to compute all
            eigenvectors of an adjacency matrix.
        device: string. optional 
            The device where the model is running on. You can specified `CPU` or `GPU` 
            for the model. (default: :str: `CPU:0`, i.e., running on the 0-th `CPU`)
        seed: interger scalar. optional 
            Used in combination with `tf.random.set_seed` & `np.random.seed` 
            & `random.seed` to create a reproducible sequence of tensors across 
            multiple calls. (default :obj: `None`, i.e., using random seed)
        name: string. optional
            Specified name for the model. (default: :str: `class.__name__`)
        kwargs: other custom keyword parameters.
        """        

        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        self.adj_transform = F.get(adj_transform)
        self.attr_transform = F.get(attr_transform)
        self.k = k
        self.process()

    def process_step(self, re_decompose=False):
        """
        Perform an adjacency matrix.

        Args:
            self: (todo): write your description
            re_decompose: (bool): write your description
        """
        graph = self.graph
        adj_matrix = self.adj_transform(graph.adj_matrix)
        attr_matrix = self.attr_transform(graph.attr_matrix)

        if re_decompose or not hasattr(self, "U"):
            V, U = sp.linalg.eigs(adj_matrix.astype('float64'), k=self.k)
            U, V = U.real, V.real
        else:
            U, V = self.U, self.V
            
        adj_matrix = (U*V) @ U.T
        adj_matrix = self.adj_transform(adj_matrix)
        
        with tf.device(self.device):
            self.feature_inputs, self.structure_inputs, self.U, self.V = F.astensors(
                attr_matrix, adj_matrix, U, V, device=self.device)            
          
    # use decorator to make sure all list arguments have the same length
    @F.EqualVarLength()        
    def build(self, hiddens=[32], activations=['relu'], dropout=0.5, l2_norm=5e-4,
              lr=0.01, use_bias=False, eps1=0.3, eps2=1.2, lamb1=0.8, lamb2=0.8):
        """
        Builds the graph.

        Args:
            self: (todo): write your description
            hiddens: (int): write your description
            activations: (todo): write your description
            dropout: (bool): write your description
            l2_norm: (todo): write your description
            lr: (todo): write your description
            use_bias: (bool): write your description
            eps1: (float): write your description
            eps2: (float): write your description
            lamb1: (todo): write your description
            lamb2: (todo): write your description
        """

        with tf.device(self.device):

            x = Input(batch_shape=[None, self.graph.n_attrs], dtype=self.floatx, name='features')
            adj = Input(batch_shape=[None, None], dtype=self.floatx, name='adj_matrix')
            index = Input(batch_shape=[None], dtype=self.intx, name='index')

            h = x
            for hid, activation in zip(hiddens, activations):
                h = DenseConvolution(hid, use_bias=use_bias,
                                     activation=activation,
                                     kernel_regularizer=regularizers.l2(l2_norm))([h, adj])

                h = Dropout(rate=dropout)(h)

            h = DenseConvolution(self.graph.n_classes, use_bias=use_bias)([h, adj])
            h = Gather()([h, index])

            model = Model(inputs=[x, adj, index], outputs=h)
            model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                          optimizer=Adam(lr=lr), metrics=['accuracy'])
            
            self.eps1 = eps1
            self.eps2 = eps2
            self.lamb1 = lamb1
            self.lamb2 = lamb2        
            self.model = model
            
    @tf.function
    def train_step(self, sequence):
        """
        Train a single model.

        Args:
            self: (todo): write your description
            sequence: (todo): write your description
        """
        (x_norm, A, idx), y = next(iter(sequence))
        
        U, V = self.U, self.V
        model = self.model
        loss_fn = model.loss
        metric = model.metrics[0]
        optimizer = model.optimizer
        model.reset_metrics()
        
        with tf.GradientTape() as tape:
            tape.watch([U, V])
            A0 = (U*V) @ tf.transpose(U)
            output = model([x_norm, A0, idx])
            loss = loss_fn(y, output)

        U_grad, V_grad = tape.gradient(loss, [U, V])
        U_grad = self.eps1*U_grad/tf.norm(U_grad)
        V_grad = self.eps2*V_grad/tf.norm(V_grad)
        
        U_hat = U + U_grad
        V_hat = V + V_grad

        with tf.GradientTape() as tape:
            A1 = (U_hat*V) @ tf.transpose(U_hat)
            A2 = (U*V_hat) @ tf.transpose(U)
            
            output0 = model([x_norm, A0, idx])
            output1 = model([x_norm, A1, idx])
            output2 = model([x_norm, A2, idx])

            loss = loss_fn(y, output0) + tf.reduce_sum(model.losses)
            loss += self.lamb1*loss_fn(y, output1) + self.lamb2*loss_fn(y, output2)
            metric.update_state(y, output0)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, metric.result()
    
    def train_sequence(self, index):
        """
        Train a tf.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        labels = self.graph.labels[index]
        with tf.device(self.device):
            sequence = FullBatchNodeSequence([self.feature_inputs, 
                                              self.structure_inputs, 
                                              index], labels)
        return sequence
