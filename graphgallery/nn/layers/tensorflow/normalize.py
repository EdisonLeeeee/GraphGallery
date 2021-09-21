import tensorflow as tf
from tensorflow.keras.layers import Layer
from graphgallery import floatx, intx


class NormalizeLayer(Layer):
    """
        Normalize the adjacency matrix with the input (edge_index, edge_weight),
        i.e., `A_hat = D^(-0.5) (A+I) D^(-0.5)`.
        refer to https://github.com/CrawlScript/tf_geometric,
        and it is deprecated because we use SparseTensor `adj` instead.
    """

    def __init__(self, rate, fill_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.fill_weight = fill_weight

    def call(self, inputs):
        edge_index, edge_weight = inputs
        num_nodes = tf.reduce_max(edge_index) + 1
        if not edge_weight:
            edge_weight = tf.ones([edge_index.shape[0]], dtype=floatx())

        edge_index, edge_weight = self.add_selfloops_edge(
            edge_index, num_nodes, edge_weight=edge_weight, fill_weight=self.fill_weight)

        row = tf.gather(edge_index, 0, axis=1)
        col = tf.gather(edge_index, 1, axis=1)
        deg = tf.math.unsorted_segment_sum(
            edge_weight, row, num_segments=num_nodes)
        deg_inv_sqrt = tf.pow(deg, self.rate)
        deg_inv_sqrt = tf.where(tf.math.is_inf(
            deg_inv_sqrt), tf.zeros_like(deg_inv_sqrt), deg_inv_sqrt)
        deg_inv_sqrt = tf.where(tf.math.is_nan(
            deg_inv_sqrt), tf.zeros_like(deg_inv_sqrt), deg_inv_sqrt)

        noremd_edge_weight = tf.gather(
            deg_inv_sqrt, row) * edge_weight * tf.gather(deg_inv_sqrt, col)

        return edge_index, noremd_edge_weight

    @staticmethod
    def add_selfloops_edge(edge_index, num_nodes, edge_weight=None, fill_weight=1.0):
        diagnal_edge_index = tf.reshape(
            tf.repeat(tf.range(num_nodes, dtype=intx()), 2), [num_nodes, 2])

        updated_edge_index = tf.concat(
            [edge_index, diagnal_edge_index], axis=0)

        if edge_weight:
            diagnal_edge_weight = tf.cast(
                tf.fill([num_nodes], fill_weight), dtype=floatx())
            updated_edge_weight = tf.concat(
                [edge_weight, diagnal_edge_weight], axis=0)

        else:
            updated_edge_weight = None

        return updated_edge_index, updated_edge_weight

    def get_config(self):
        config = {'rate': self.rate, 'fill_weight': self.fill_weight}

        base_config = super().get_config()
        return {**base_config, **config}
