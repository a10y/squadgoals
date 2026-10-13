"""
Simple utility functions that don't necessarily fit elsewhere.
"""

import time
import math
from functools import wraps

import numpy as np
import tensorflow as tf

"""
Assertions
"""
def assert_rank(name, tensor, expected_rank):
    actual_rank = len(tensor.get_shape())
    assert expected_rank == actual_rank, \
        "Wrong shape for {}! Expected {}, Actual {}".format(name, expected_rank, actual_rank)

def assert_dim(name, tensor, dim, expected_value):
    actual_shape = tensor.get_shape()
    actual_value = actual_shape[dim]
    assert expected_value == actual_value, \
        "Wrong shape for {}! Expected shape[{}] == {}, was {}. Total shape: {}".format(
            name, dim, expected_value, actual_value, actual_shape
        )


def load_glove():
    try:
        return np.load("data/squad/glove.trimmed.100.npz")["glove"]
    except IOError as e:
        print("Error loading GloVe data: {}".format(e))


def minibatch_index_iterator(total_size, batch_size):
    """
    Generator that yields indexes to sample the minibatch from up to the total size.
    """
    order = np.random.permutation(total_size)
    for i in range(0, order.shape[0], batch_size):
        yield order[i:i + batch_size]

def compute_once(expensive):
    """
    Decorator for once-evaluated function. Python 3.x only (use of 'nonlocal' keyword)
    """
    computed = False
    value = None
    @wraps(expensive)
    def inner(*args, **kwargs):
        nonlocal computed
        nonlocal value
        if not computed:
            value = expensive(*args, **kwargs)
            computed = True
        return value
    return inner

def batch_matmul(xs, W):
    shape = tf.shape(xs)
    W_shape = tf.shape(W)
    xs = tf.reshape(xs, [-1, shape[2]])
    result = tf.matmul(xs, W)
    return tf.reshape(result, [shape[0], shape[1], W_shape[1]]) # Final dimension should be last dim of W


class Progress(object):
    def __init__(self, title="", width=30, steps='unknown', sameline=True):
        self._title = title
        self._width = width
        self._steps = steps
        self._pos = -1
        self._last_tick = None
        self._sameline = sameline

    def tick(self, **kwargs):
        self._pos += 1
        title = "{} ({} of {}):  ".format(self._title, self._pos, self._steps)

        # Provide estimate of time between ticks
        now = time.time()
        if self._last_tick is not None:
            delta = now - self._last_tick
            eta = (self._steps - self._pos) * delta
            kwargs['time'] = "{:.2f}s".format(delta)
            kwargs['hz'] = "{:.2f}it/s".format(1.0 / delta)
            kwargs['eta'] = "{:02d}:{:02d}".format(math.floor(eta / 60), int(eta % 60))
        self._last_tick = now

        pairs = []
        for k, v in kwargs.items():
            # Truncate floats
            if isinstance(v, (float, np.floating)):
                v = "{:.4f}".format(v)
            pairs.append("{}={}".format(k, v))
        extra_info = '   '.join(pairs)

        ending = "\r" if self._sameline else "\n"
        if ending == "\r":
            print("\r" +  " " * 200, end=ending)
        text = title + "\t" + extra_info
        print(text, end=ending)

        if self._pos == self._steps:
            print()
