# coding: utf-8
import tensorflow as tf

def variable_summaries(var, name, mean=True, sigma=True, scope=None):
    '''
    var: variable
    '''
    with tf.variable_scope(scope or 'summaries') as sc:
        tf.summary.histogram(name, var)

        if mean:
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)

        if sigma:
            sigma = tf.reduce_mean(tf.square(var - mean))
            tf.summary.scalar('sigma/' + name, sigma)
        return sc




