import tensorflow as tf


def get_normalized_vector(d):
    d /= (1e-12 + tf.reduce_max(tf.abs(d), keepdims=True))
    d /= tf.math.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), axis=-1, keepdims=True))
    return d


def kl_divergence_with_logit(q_logit, p_logit,
                             mask=None):
    q = tf.math.softmax(q_logit)
    qlogp = tf.reduce_sum(q * tf.math.log_softmax(p_logit), axis=-1)
    if mask is not None:
        mask /= tf.reduce_mean(mask)
        qlogp *= mask

    return -tf.reduce_mean(qlogp)


def entropy_y_x(logit):
    p = tf.math.softmax(logit)
    return -tf.reduce_mean(tf.reduce_sum(p * tf.math.log_softmax(logit), axis=-1))
