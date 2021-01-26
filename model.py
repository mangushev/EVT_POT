
#TODO:

import tensorflow as tf

import sys
import math
import six

import numpy as np

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape

def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)

def print_shape(tensor, rank, tensor_name):
  #return tensor
  tensor_shape = get_shape_list(tensor, expected_rank=rank)
  return tf.Print(tensor, [tensor_shape], tensor_name, summarize=8)

class POT(object):
  #   A - batch size

  #   Sigma = scale parameter of GPD
  #   Gamma = shape parameter of GPD
  #   l - losses
  #   t - initial threshold
  #   n - number of observer values
  #   Nt = number of exeedances
  #   q = proportion of exceedances. If manually set : 0.001. q is a probability that we will not exceed threshold Zq 

  def __init__(self,
               exceedances,
               n,
               t,
               q,
               initializer_range):

    exceedances = tf.reshape(exceedances, [-1])

    epsilon = tf.constant(1e-14, dtype=tf.float32)

    #self._sigma = tf.compat.v1.get_variable(initializer=tf.random_uniform_initializer(minval=0.167, maxval=0.169), shape=[], trainable=True, name='sigma')
    self._sigma = tf.compat.v1.get_variable(initializer=tf.random_uniform_initializer(minval=1, maxval=2), shape=[], trainable=True, name='sigma')
    #self._gamma = tf.compat.v1.get_variable(initializer=tf.random_uniform_initializer(minval=-0.009, maxval=-0.01), shape=[], trainable=True, name='gamma')
    self._gamma = tf.compat.v1.get_variable(initializer=tf.random_uniform_initializer(minval=1, maxval=2), shape=[], trainable=True, name='gamma')
 
    Nt = tf.cast(tf.size(exceedances), dtype=tf.float32)

    self._loss = -(-Nt*tf.math.log(self._sigma+epsilon)-(1 + 1/(self._gamma+epsilon))*tf.reduce_sum(tf.math.log(1+(self._gamma/(self._sigma+epsilon))*exceedances)))

    self._threshold = t + (self._sigma / (self._gamma+epsilon)) * (tf.math.pow((q * n) / Nt, -self._gamma) - 1)   

  @property
  def loss(self):
    return self._loss

  @property
  def sigma(self):
    return tf.reshape(self._sigma, [1])

  @property
  def gamma(self):
    return tf.reshape(self._gamma, [1])

  @property
  def threshold(self):
    return tf.reshape(self._threshold, [1])
