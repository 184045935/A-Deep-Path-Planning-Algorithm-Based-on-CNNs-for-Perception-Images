# coding: utf-8
import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer
from tensorflow.python.ops.init_ops import VarianceScaling

# initializer default to glorot_uniform_initializer

__all__ = ['glorot_uniform',
           'xavier_uniform',
           'glorot_normal',
           'xavier_normal',
           'he_normal',
           'he_uniform',
           'preset_initializers']

preset_initializers = ['glorot_uniform',
                       'xavier_uniform',
                       'glorot_normal',
                       'xavier_normal',
                       'he_normal',
                       'he_uniform']

def glorot_uniform(seed=None, dtype=tf.float32):
    return VarianceScaling(scale=1.0, 
                           mode='fan_avg', 
                           distribution='uniform', 
                           seed=seed,
                           dtype=tf.float32)

def glorot_normal(seed=None, dtype=tf.float32):
    return VarianceScaling(scale=1.0, 
                           mode='fan_avg', 
                           distribution='normal', 
                           seed=seed,
                           dtype=tf.float32)

xavier_uniform = glorot_uniform
xavier_normal = glorot_normal

def he_normal(seed=None, mode='fan_in', dtype=tf.float32):
    '''
    mode: a string. 'fan_in', 'fan_out' or 'fan_avg'
    '''
    return VarianceScaling(scale=2.0,
                        mode=mode,
                        distribution='normal',
                        seed=seed,
                        dtype=tf.float32)

def he_uniform(seed=None, mode='fan_in', dtype=tf.float32):
    '''
    mode: a string. 'fan_in', 'fan_out' or 'fan_avg'
    '''
    return VarianceScaling(scale=2.0,
                           mode=mode,
                           distribution='uniform',
                           seed=seed,
                           dtype=tf.float32)