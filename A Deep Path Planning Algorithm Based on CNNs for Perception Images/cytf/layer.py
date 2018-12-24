# coding: utf-8
import tensorflow as tf
from cytf.initializers import *
from cytf.activations import *
from cytf.tfboard_utils import variable_summaries
from cytf.arg_scope import *
from cytf.utils import *
from tensorflow.contrib.layers.python.layers import utils
import sys

@add_arg_scope
def conv2d(input, num_outputs, kernel_size, stride=1, padding='SAME', initializer=None, name='conv2',
           activation=None, regualizer=None, TFboard_recording=None, train=True, BN=None, 
           ema=False, ema_op=None
          ):
    '''
    name: 
        suggested name format: 'layer_n/conv2_n' if without variable_scope('layer_n') outside
    kernel_size: 
        a list of length 2: [kernel_height, kernel_width]
    stride: 
        a list of length 2: [stride_height, stride_width]. Can be an int if both values are the same.
    padding: 
        default to 'SAME', also can be 'VALID'
    regualizer: 
        regualizer added to weights. format:{'l2': 0.004}. can be 'l1', 'l2' or 'l1_l2'
    initializer: 
        a string or a self-defined initializer. Default to glorot_uniform. 
    ema: 
        whether use ema (exponential moving average). If True, add the weights and biases param to 
        collection "ema" in the training process. Default to False.
    ema_op: 
        the ema_op used in the evaluation process.
    BN: 
        whether to use batch norm. Param decay is set to 0.9 by default. Can use a float number to explicitly
        set this param.
    train: 
        whether it's a training procedure or not. Default to True.
    activation:
        whether to apply activation function. Now only support string form. To be finished.
    TFboard_recording:
        a list. Format: [1, 1, 1, 1, 1]
        weights, biases, Wx_plus_b, after_BN, after_activaition
    '''
    input_shape = input.get_shape().as_list()
    num_input = input_shape[-1]
    layer_name = name.split('/')[0]
    summary_scope = None

    #TODO: give an alert to TFboard_recording when without activation

    # strides format setting
    if isinstance(stride, int):
        strides = [1, stride, stride, 1]
    elif isinstance(stride, list):
        if len(stride) != 2:
            raise ValueError('In layer "%s": len(stride) must be 2 when stride is a list!' % name)
        strides = [1, stride[0], stride[1], 1]
    else:
        raise ValueError('In layer "%s": param stride must be an int or a list!' % name)

    # kernel_shape format setting
    kernel_shape = kernel_size + [num_input, num_outputs]

    # initializer setting
    if not initializer:
        initializer = glorot_uniform
    elif initializer in preset_initializers:
        initializer = getattr(sys.modules[__name__], initializer)
    # self_defined initializer
    # e.g: initializer = he_normal(mode='fan_out)
    else:
        initializer = initializer

    with tf.variable_scope(name):
            
        if ema_op:
            W, b = ema_op.average([
                                   tf.get_collection('variables', name+'/weights:0')[0],
                                   tf.get_collection('variables', name+'/biases:0')[0]
                                   ])
        else:
            W = tf.get_variable(name='weights', shape=kernel_shape, initializer=initializer())
            b = tf.get_variable(name='biases', shape=[num_outputs], initializer=tf.zeros_initializer())
            # 只执行一次
            if ema:
                tf.add_to_collection('ema', W)
                tf.add_to_collection('ema', b)
        
        if TFboard_recording and TFboard_recording[0]:
            if not summary_scope:
                summary_scope = variable_summaries(W, 'W')
            else:
                # .original_name_scope has the nested scope name form
                variable_summaries(W, 'W', scope=summary_scope.original_name_scope)
        if TFboard_recording and TFboard_recording[1]:
            if not summary_scope:
                summary_scope = variable_summaries(b, 'b')
            else:
                variable_summaries(b, 'b', scope=summary_scope.original_name_scope)

        net = tf.identity(tf.nn.conv2d(input, W, strides=strides, padding=padding.upper()) + b,
                        name='Wx_plus_b')
                        
        if TFboard_recording and TFboard_recording[2]:
            if not summary_scope:
                summary_scope = variable_summaries(net, 'Wx_plus_b')
            else:
                variable_summaries(net, 'Wx_plus_b', scope=summary_scope.original_name_scope)

        if BN:
            if isinstance(BN, bool):
                bn = batch_norm()
            elif isinstance(BN, float):
                bn = batch_norm(decay=BN)
            else:
                raise ValueError('In layer "%s": param BN must be a bool or a float number!' % name)
            net = bn(net, train=train)
            if TFboard_recording and TFboard_recording[3]:
                if not summary_scope:
                    summary_scope = variable_summaries(net, 'BN')
                else:
                    variable_summaries(net, 'BN', scope=summary_scope.original_name_scope)

        if regualizer:
            regualizer_key= list(regualizer.keys())[0]
            if regualizer_key not in ['l1', 'l2', 'l1_l2']:
                raise ValueError('In layer "%s": regualizer_key must be "l1", "l2" or "l1_l2"!' % name)
            regualizer_name = regualizer_key + '_regularizer'
            regualizer_lambda = regualizer[regualizer_key]
            if regualizer_key in ['l1', 'l2']:
                tf.add_to_collection('losses', 
                                    getattr(tf.contrib.layers, regualizer_name)(regualizer_lambda)(W))
            # l1_l2 loss
            else:
                tf.add_to_collection('losses', 
                                    getattr(tf.contrib.layers, regualizer_name)(*regualizer_lambda)(W))
    
    # activation function setting
    if activation in preset_activations:
        with tf.name_scope(layer_name + '/' + activation):
            activation_ = getattr(sys.modules[__name__], activation)
            net = activation_(net, name=activation)
        
        with tf.variable_scope(name):
            if TFboard_recording and TFboard_recording[4]:
                if not summary_scope:
                    summary_scope = variable_summaries(net, 'after_activaition')
                else:
                    variable_summaries(net, 'after_activaition', scope=summary_scope.original_name_scope)

    return net

@add_arg_scope
def dense(input, num_outputs, name='dense', initializer=None, ema=False, ema_op=None, TFboard_recording=None,
          BN=False, regualizer=None, train=True, activation=None):
    '''
    TFboard_recording: 
        [1, 1, 1, 1, 1]
        weights, biases, Wx_plus_b, after_BN, after_activaition
    '''
    input_shape = input.get_shape().as_list()
    num_input = input_shape[-1]
    layer_name = name.split('/')[0]
    summary_scope = None

    # initializer setting
    if not initializer:
        initializer = glorot_uniform
    elif initializer in preset_initializers:
        initializer = getattr(sys.modules[__name__], initializer)
    # self_defined initializer
    # e.g: initializer = he_normal(mode='fan_out)
    else:
        initializer = initializer
    
    with tf.variable_scope(name):
        if ema_op:
            W, b = ema_op.average([
                                   tf.get_collection('variables', name+'/weights:0')[0],
                                   tf.get_collection('variables', name+'/biases:0')[0]
                                   ])
        else:
            W = tf.get_variable(name='weights', shape=[num_input, num_outputs], initializer=initializer())
            b = tf.get_variable(name='biases', shape=[num_outputs], initializer=tf.zeros_initializer())

            if ema:
                tf.add_to_collection('ema', W)
                tf.add_to_collection('ema', b)
        
        if TFboard_recording and TFboard_recording[0]:
            if not summary_scope:
                summary_scope = variable_summaries(W, 'W')
            else:
                variable_summaries(W, 'W', scope=summary_scope.original_name_scope)
        if TFboard_recording and TFboard_recording[1]:
            if not summary_scope:
                summary_scope = variable_summaries(b, 'b')
            else:
                variable_summaries(b, 'b', scope=summary_scope.original_name_scope)
        
        net = tf.identity(tf.matmul(input, W) + b, name='Wx_plus_b')
        if TFboard_recording and TFboard_recording[2]:
            if not summary_scope:
                summary_scope = variable_summaries(net, 'Wx_plus_b')
            else:
                variable_summaries(net, 'Wx_plus_b', scope=summary_scope.original_name_scope)
        
        if BN:
            if isinstance(BN, bool):
                bn = batch_norm()
            elif isinstance(BN, float):
                bn = batch_norm(decay=BN)
            else:
                raise ValueError('In layer "%s": param BN must be a bool or a float number!' % name)
            net = bn(net, train=train)
            if TFboard_recording and TFboard_recording[3]:
                if not summary_scope:
                    summary_scope = variable_summaries(net, 'BN')
                else:
                    variable_summaries(net, 'BN', scope=summary_scope.original_name_scope)
        
        if regualizer:
            regualizer_key= list(regualizer.keys())[0]
            if regualizer_key not in ['l1', 'l2', 'l1_l2']:
                raise ValueError('In layer "%s": regualizer_key must be "l1", "l2" or "l1_l2"!' % name)
            regualizer_name = regualizer_key + '_regularizer'
            regualizer_lambda = regualizer[regualizer_key]
            if regualizer_key in ['l1', 'l2']:
                tf.add_to_collection('losses', 
                                    getattr(tf.contrib.layers, regualizer_name)(regualizer_lambda)(W))
            # l1_l2 loss
            else:
                tf.add_to_collection('losses', 
                                    getattr(tf.contrib.layers, regualizer_name)(*regualizer_lambda)(W))
            
    # activation function setting
    if activation in preset_activations:
        with tf.name_scope(layer_name + '/' + activation):
            activation_ = getattr(sys.modules[__name__], activation)
            net = activation_(net, name=activation)
        
        with tf.variable_scope(name):
            if TFboard_recording and TFboard_recording[4]:
                if not summary_scope:
                    summary_scope = variable_summaries(net, 'after_activaition')
                else:
                    variable_summaries(net, 'after_activaition', scope=summary_scope.original_name_scope)

    return net    

@add_arg_scope
def flatten(input, name='flatten'):
    input_shape = input.get_shape().as_list()

    if len(input_shape) < 4:
        raise ValueError('In layer "%s": the length of the input must be greater that 4!' % name)
    else:
        output_length = 1
        for dim in input_shape[1:]:
            output_length *= dim

    with tf.variable_scope(name):
        net = tf.reshape(input, [-1, output_length])
        return net

@add_arg_scope
def max_pool(input, ksize, stride=2, padding='VALID', name='maxpool'):
    '''
    ksize:
        Pooling window size. A list of length 2: [window_height, window_width]
    stride: 
        A list of length 2: [stride_height, stride_width]. Can be an int if both values are the same.
    padding:
        default to 'VALID', also can be 'SAME'.
    '''
    kernel_size = [1] + ksize + [1]
    if isinstance(stride, int):
        strides = [1, stride, stride, 1]
    elif isinstance(stride, list):
        if len(stride) != 2:
            raise ValueError('In layer "%s": len(stride) must be 2 when stride is a list!' % name)
        strides = [1, stride[0], stride[1], 1]
    
    with tf.variable_scope(name):
        net = tf.nn.max_pool(input, ksize=kernel_size, strides=strides, padding=padding.upper())
        return net

@add_arg_scope
def activation(input, activation_func=None, name='activation', TFboard_recording=False):
    '''
    activation_func:
        Can be a string from preset_activations. Or a specific activation function
    TFboard_recording:
        True or False. Default to False
    '''
    with tf.variable_scope(name):
        if activation_func in preset_activations:
            activation_func_ = getattr(sys.modules[__name__], activation_func)
            net = activation_func_(input, name=activation_func)
        else:
            net = activation_func(input)
        
        if TFboard_recording:
            variable_summaries(net, 'after_activation')
        
        return net

@add_arg_scope
def dropout(input, keep_prob=None, name='dropout'):
    with tf.variable_scope(name):
        net = tf.nn.dropout(input, keep_prob=keep_prob, name=name)
        return net

class batch_norm(object):

    def __init__(self, decay=0.9):
        self.epsilon = 1e-5  # use numbers closer to 1 if you have more data
        self.decay = decay

        self.ema = EMA(decay=self.decay, scope_suffix='moving_average')


    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        with tf.variable_scope('batch_norm') as scope:
            
            beta = tf.get_variable("beta", [shape[-1]],
                                            initializer=tf.constant_initializer(0.))
            # gamma = tf.get_variable("gamma", [shape[-1]],
            #                                  initializer=tf.constant_initializer(1.))
            gamma = tf.get_variable("gamma", [shape[-1]],
                                             initializer=tf.random_normal_initializer(mean=1.0, stddev=0.002))

            def batch_norm_training():

                input_ndim = len(shape)
                axis = list(range(input_ndim - 1))

                self.batch_mean, self.batch_var = tf.nn.moments(x, axis, name='moments')

                ema_apply_op = self.ema.apply([self.batch_mean, self.batch_var], ['mean', 'variance'])

                # compute the ema first
                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(self.batch_mean), tf.identity(self.batch_var)
                return mean, var

            def batch_norm_inference():
                mean, var = self.ema.average(self.batch_mean), self.ema.average(self.batch_var)
                return mean, var

            train = tf.convert_to_tensor(train)
            mean, var = tf.cond(tf.equal(train, tf.constant(True)), batch_norm_training, batch_norm_inference)

            normed = tf.nn.batch_normalization(
            x, mean, var, beta, gamma, self.epsilon, name='compute')

            return normed