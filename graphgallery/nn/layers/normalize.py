import tensorflow as tf
from tensorflow.keras.layers import Layer
from graphgallery import config


class NormalizeLayer(Layer):
    """
        Normalize the adjacency matrix with the input (edge_index, edge_weight),
        i.e., `A_hat = D^(-0.5) (A+I) D^(-0.5)`.
        refer to https://github.com/CrawlScript/tf_geometric,
        and it is deprecated because we use SparseTensor `adj` instead.
    """

    def __init__(self, norm_adj, **kwargs):
        super().__init__(**kwargs)
        self.norm_adj = norm_adj

    def call(self, inputs, improved=False):
        edge_index, edge_weight = inputs
        n_nodes = tf.reduce_max(edge_index) + 1
        if not edge_weight:
            edge_weight = tf.ones([edge_index.shape[0]], dtype=config.floatx())

        fill_weight = 2.0 if improved else 1.0
        edge_index, edge_weight = self.add_self_loop_edge(edge_index, n_nodes, edge_weight=edge_weight, fill_weight=fill_weight)

        row = tf.gather(edge_index, 0, axis=1)
        col = tf.gather(edge_index, 1, axis=1)
        deg = tf.math.unsorted_segment_sum(edge_weight, row, num_segments=n_nodes)
        deg_inv_sqrt = tf.pow(deg, self.norm_adj)
        deg_inv_sqrt = tf.where(tf.math.is_inf(deg_inv_sqrt), tf.zeros_like(deg_inv_sqrt), deg_inv_sqrt)
        deg_inv_sqrt = tf.where(tf.math.is_nan(deg_inv_sqrt), tf.zeros_like(deg_inv_sqrt), deg_inv_sqrt)

        noremd_edge_weight = tf.gather(deg_inv_sqrt, row) * edge_weight * tf.gather(deg_inv_sqrt, col)

        return edge_index, noremd_edge_weight

    @staticmethod
    def add_self_loop_edge(edge_index, n_nodes, edge_weight=None, fill_weight=1.0):
        diagnal_edge_index = tf.reshape(tf.repeat(tf.range(n_nodes, dtype=config.intx()), 2), [n_nodes, 2])

        updated_edge_index = tf.concat([edge_index, diagnal_edge_index], axis=0)

        if edge_weight:
            diagnal_edge_weight = tf.cast(tf.fill([n_nodes], fill_weight), dtype=config.floatx())
            updated_edge_weight = tf.concat([edge_weight, diagnal_edge_weight], axis=0)

        else:
            updated_edge_weight = None

        return updated_edge_index, updated_edge_weight

    def get_config(self):
        config = {'norm_adj': self.norm_adj}

        base_config = super().get_config()
        return {**base_config, **config}
