# coding: utf-8

import tensorflow as tf

from tensorflow.python.training import moving_averages
from tensorflow.python.framework import ops

__all__ = ['EMA']

class EMA(object):
    """
    Calculate exponential Moving Average. Implemented by wizyoung.
    Pros:
        the exactly same usage with the builtin `tf.train.ExponentialMovingAverage` api.
        no duplicate scope suffix caused by `_create_slot_var` method.
        can specify the name of shadow variables by hand.
    You can check the differences in `tensorflow/python/training/moving_averages.py`.

    TODO: not yet finished `variables_to_restore` function and `num_updates` keyword
    """
    def __init__(self, decay, zero_debias=False, scope_suffix='ema'):
        """
        
        Args:
            decay (float): The decay to use.
            zero_debias (bool, optional): Defaults to False. If `True`, zero debias moving-averages that are initialized
                with tensors.
            scope_suffix (str, optional): Defaults to 'ema'. The scope_suffix of the shadow variables.
        """

        self._decay = ops.convert_to_tensor(decay, name='decay')
        self._averages = {}
        self._zero_debias = zero_debias
        self._scope_suffix = scope_suffix + '_'

    def apply(self, varlist, namelist=None):
        """
        Maintains moving averages of variables.
        
        Args:
            varlist (list): The list of vars to maintain the moving average. Can be Variables or Tensor objects.
            namelist (list, optional): Defaults to None. The name added to the scope_suffix forming the shadow Variable name, like
                'ema_mean'. If not specified, the original var name will be directly used.
        """

        zero_debias_true = set()  # set of vars to set `zero_debias=True` (for tensor instead of variable)
        if namelist:
            if not len(varlist) == len(namelist):
                raise ValueError("len(varlist) must be equal to len(namelist) When specifing namelist in EMA apply function!")
        else:
            namelist = [None] * len(varlist)
        for (var, name) in zip(varlist, namelist):
            var_name = name or var.name.split(':')[0].split('/')[-1]
            if var in self._averages:
                raise ValueError("Moving average already computed for: %s" % var.name)
            # variable condition
            if isinstance(var, tf.Variable):
                avg = tf.get_variable(initializer=var.initialized_value(), name=self._scope_suffix + var_name)
                tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)
            # tensor condition
            else:
                avg = tf.get_variable(shape=var.shape, initializer=tf.zeros_initializer(), name=self._scope_suffix + var_name)
                # zero_debias
                if self._zero_debias:
                    zero_debias_true.add(avg)
            self._averages[var] = avg

        updates = []
        for var in varlist:
            zero_debias = self._averages[var] in zero_debias_true
            updates.append(moving_averages.assign_moving_average(
            self._averages[var], var, self._decay, zero_debias=zero_debias))
        return tf.group(*updates)

    def average(self, var):
        """
        Returns the `Variable` holding the average of `var`
        
        Args:
            var : A `Variable` object.
        
        Returns:
            A `Variable` object or `None` if the moving average of `var`
            is not maintained.
        """
        return self._averages.get(var, None)
    
    def average_name(self, var):
        """
        Returns the name of the `Variable` holding the average for `var`
        """

        if var in self._averages:
            return self._averages[var].op.name