import numpy as np
import numpy.random as rd
import tensorflow as tf



def average_w_matrix(w, factor):
    """ Performs average operation on matrix with kernel of size [1,'factor'] , with stride = 'factor' """

    if factor == 1:
        return w

    w = tf.expand_dims(tf.constant(w), axis=0)
    w = tf.expand_dims(w, axis=-1)

    filter_avg = tf.squeeze(tf.stack([tf.ones([1, factor]) / factor, tf.zeros([1, factor])], axis=1))
    # filter_avg = tf.squeeze(tf.stack([tf.ones([1, factor]), tf.zeros([1, factor])], axis=1))
    filter_avg = tf.expand_dims(tf.expand_dims(filter_avg, axis=-1), axis=-1)

    return tf.squeeze(tf.nn.conv2d(w, filter_avg, strides=[1, factor], padding="SAME"))

