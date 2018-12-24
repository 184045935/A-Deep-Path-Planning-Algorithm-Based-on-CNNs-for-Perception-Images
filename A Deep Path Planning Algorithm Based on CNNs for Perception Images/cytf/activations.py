# coding: utf-8
import tensorflow as tf

__all__ = ['relu',
        #    'leaky_relu',
           'sigmoid',
           'tanh',
           'preset_activations']

# aliases
relu = tf.nn.relu
sigmoid = tf.nn.sigmoid
tanh = tf.nn.tanh
# leaky_relu = tf.nn.leaky_relu

preset_activations = ['relu',
                    #   'leaky_relu',
                      'sigmoid',
                      'tanh']


