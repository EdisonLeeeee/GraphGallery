"""
util functions for 'Batch Virtual Adversarial Training' (BVAT)
"""
import tensorflow as tf
from graphgallery.typing import TFTensor


def get_normalized_vector(d: TFTensor) -> TFTensor:
    d /= (1e-12 + tf.reduce_max(tf.abs(d)))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), 1, keepdims=True))
    return d


def kl_divergence_with_logit(q_logit: TFTensor, p_logit: TFTensor,
                             mask: TFTensor = None) -> TFTensor:
    q = tf.math.softmax(q_logit)
    if mask is None:
        qlogp = tf.reduce_mean(tf.reduce_sum(q * tf.math.log_softmax(p_logit), 1))
    else:
        mask /= tf.reduce_mean(mask)
        qlogp = tf.reduce_mean(tf.reduce_sum(q * tf.math.log_softmax(p_logit), 1) * mask)

    return -qlogp


def entropy_y_x(logit: TFTensor) -> TFTensor:
    p = tf.math.softmax(logit)
    return -tf.reduce_mean(tf.reduce_sum(p * tf.math.log_softmax(logit), 1))
