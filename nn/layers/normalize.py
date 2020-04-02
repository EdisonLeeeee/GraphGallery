import tensorflow as tf
from tensorflow.keras.layers import Layer


class NormalizeLayer(Layer):
    '''deprecated'''
    
    def __init__(self, n_nodes, normalize_rate, **kwargs):
        super().__init__(**kwargs)
        self.n_nodes = n_nodes
        self.normalize_rate = normalize_rate
        

    def call(self, inputs, improved=False):
        edge_index, edge_weight = inputs
        if edge_weight is None:
            edge_weight = tf.ones([edge_index.shape[0]], dtype=tf.float32)

        fill_weight = 2.0 if improved else 1.0
        edge_index, edge_weight = self.add_self_loop_edge(edge_index, self.n_nodes, edge_weight=edge_weight, fill_weight=fill_weight)
        
        row = tf.gather(edge_index, 0, axis=1)
        col = tf.gather(edge_index, 1, axis=1)
        deg = tf.math.unsorted_segment_sum(edge_weight, row, num_segments=self.n_nodes)
        deg_inv_sqrt = tf.pow(deg, self.normalize_rate)
        deg_inv_sqrt = tf.where(tf.math.is_inf(deg_inv_sqrt), tf.zeros_like(deg_inv_sqrt), deg_inv_sqrt)
        deg_inv_sqrt = tf.where(tf.math.is_nan(deg_inv_sqrt), tf.zeros_like(deg_inv_sqrt), deg_inv_sqrt)

        noremd_edge_weight = tf.gather(deg_inv_sqrt, row) * edge_weight * tf.gather(deg_inv_sqrt, col)

        return edge_index, noremd_edge_weight
    
        
    @staticmethod    
    def add_self_loop_edge(edge_index, n_nodes, edge_weight=None, fill_weight=1.0):
        diagnal_edge_index = tf.reshape(tf.repeat(tf.range(n_nodes, dtype=tf.int64), 2), [n_nodes,2])
        
        updated_edge_index = tf.concat([edge_index, diagnal_edge_index], axis=0)

        if edge_weight is not None:
            diagnal_edge_weight = tf.cast(tf.fill([n_nodes], fill_weight), tf.float32)
            updated_edge_weight = tf.concat([edge_weight, diagnal_edge_weight], axis=0)

        else:
            updated_edge_weight = None

        return updated_edge_index, updated_edge_weight
    
    
    def get_config(self):
        config = {'n_nodes': self.n_nodes,
                  'normalize_rate': self.normalize_rate,
                }
        
        base_config = super().get_config()
        return {**base_config, **config}