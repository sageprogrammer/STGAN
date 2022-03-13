from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
tf1 = tf.compat.v1


def load_checkpoint(ckpt_dir_or_file, session, var_list=None):
    """Load checkpoint.

    This function add some useless ops to the graph. It is better
    to use tf1.train.init_from_checkpoint(...).
    """
    try:
        if os.path.isdir(ckpt_dir_or_file):
            ckpt_dir_or_file = tf1.train.latest_checkpoint(ckpt_dir_or_file)
    
        restorer = tf1.train.Saver(var_list)
        restorer.restore(session, ckpt_dir_or_file)
        print(' [*] Loading checkpoint succeeds! Copy variables from % s!' % ckpt_dir_or_file)
        return True
    except:
        return False


def init_from_checkpoint(ckpt_dir_or_file, assignment_map={'/': '/'}):
    # Use the checkpoint values for the variables' initializers. Note that this
    # function just changes the initializers but does not actually run them, and
    # you should still run the initializers manually.
    tf1.train.init_from_checkpoint(ckpt_dir_or_file, assignment_map)
    print(' [*] Loading checkpoint succeeds! Copy variables from % s!' % ckpt_dir_or_file)
