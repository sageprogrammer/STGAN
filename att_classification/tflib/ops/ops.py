from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf1 = tf.compat.v1


def minmax_norm(x, epsilon=1e-12):
    x = tf1.to_float(x)
    min_val = tf1.reduce_min(x)
    max_val = tf1.reduce_max(x)
    x_norm = (x - min_val) / tf1.maximum((max_val - min_val), epsilon)
    return x_norm
