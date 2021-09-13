import numpy as np
import graphgallery as gg

__all__ = ["jaccard_similarity", "cosine_similarity",
           "kld_divergence", "neighborhood_entropy"]


def jaccard_similarity(A, B):
    intersection = np.count_nonzero(A * B, axis=1)
    J = intersection * 1.0 / (np.count_nonzero(A, axis=1) + np.count_nonzero(B, axis=1) + intersection + gg.epsilon())
    return J


def cosine_similarity(A, B):
    inner_product = (A * B).sum(1)
    C = inner_product / (np.sqrt(np.square(A).sum(1)) * np.sqrt(np.square(B).sum(1)) + gg.epsilon())
    return C


def kld_divergence(A, B):
    import tensorflow as tf
    kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
    D = kl(A, B) + kl(B, A)
    return D


def neighborhood_entropy(node_info, neighbor_infos):
    infos = (neighbor_infos.sum(0) + node_info) / (neighbor_infos.shape[0] + 1)
    entropy = infos * np.log2(infos + gg.epsilon())
    return -np.sum(entropy)
