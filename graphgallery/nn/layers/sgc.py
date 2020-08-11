import tensorflow as tf
from tensorflow.keras.layers import Layer


class SGConvolution(Layer):
    """
        Simplifying graph convolution layer as in: 
        [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153)
        Pytorch implementation: https://github.com/Tiiiger/SGC
        
        `SGConvolution` implements the operation:
        `output = x @ adj^{order}`
        where `x` is the feature matrix, `adj` is the adjacency matrix.
        
        Note:
          This `SGConvolution` layer has NOT any trainable parameters.
        
        
        Arguments:
          order: Positive integer, the power of adjacency matrix, i.e., adj^{order}.

        Input shape:
          tuple/list with two 2-D tensor: Tensor `x` and SparseTensor `adj`: `[(n_nodes, n_features), (n_nodes, n_nodes)]`.
          The former one is the feature matrix (Tensor) and the other is adjacency matrix (SparseTensor).

        Output shape:
          2-D tensor with shape: `(n_nodes, n_features)`.       
    """
    def __init__(self, order=1, **kwargs):
        super().__init__(**kwargs)
        self.order = order
        

    def call(self, inputs):
        x, adj = inputs
        
        for _ in range(self.order):
            x = tf.sparse.sparse_dense_matmul(adj, x)

        return x
    
    
    def get_config(self):
        config = {'order': self.order}
        
        base_config = super().get_config()
        return {**base_config, **config}
    
    
    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        return tf.TensorShape(features_shape) # (n_nodes, n_features)
