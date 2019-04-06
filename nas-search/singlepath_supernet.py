# This code extends codebase from the "MNasNet on TPU" GitHub repo:
# https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
#
# This project incorporates material from the project listed above, and it
# is accessible under their original license terms (Apache License 2.0)
# ==============================================================================
"""Contains the supernet definition based on the Single-Path
   search space formulation.

[1] D. Stamoulis et al., Single-Path NAS: Designing Hardware-Efficient 
    ConvNets in less than 4 Hours. arXiv:(TBD)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import json

# dstamoulis: definition of masked layer (DepthwiseConv2DMasked)
from superkernel import *

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'depth_multiplier', 'depth_divisor', 'min_depth', 'search_space',
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

# TODO(hongkuny): Consider rewrite an argument class with encoding/decoding.
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for convolutional kernels.

  The main difference with tf.variance_scaling_initializer is that
  tf.variance_scaling_initializer uses a truncated normal with an uncorrected
  standard deviation, whereas here we use a normal distribution. Similarly,
  tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
  a corrected standard deviation.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
  del partition_info
  kernel_height, kernel_width, _, out_filters = shape
  fan_out = int(kernel_height * kernel_width * out_filters)
  return tf.random_normal(
      shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for dense kernels.

  This initialization is equal to
    tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                    distribution='uniform').
  It is written out explicitly here for clarity.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
  del partition_info
  init_range = 1.0 / np.sqrt(shape[1])
  return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


def round_filters(filters, global_params):
  """Round number of filters based on depth multiplier."""
  multiplier = global_params.depth_multiplier
  divisor = global_params.depth_divisor
  min_depth = global_params.min_depth
  if not multiplier:
    return filters

  filters *= multiplier
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  return new_filters


class MBConvBlock(object):
  """A class of MnasNet/MobileNetV2 Inveretd Residual Bottleneck.

  Attributes:
    has_se: boolean. Whether the block contains a Squeeze and Excitation layer
      inside.
    endpoints: dict. A list of internal tensors.
  """

  def __init__(self, block_args, global_params, layer_runtimes, dropout_rate):
    """Initializes a MBConv block.

    Args:
      block_args: BlockArgs, arguments to create a MnasBlock.
      global_params: GlobalParams, a set of global parameters.
    """
    self._block_args = block_args
    self._batch_norm_momentum = global_params.batch_norm_momentum
    self._batch_norm_epsilon = global_params.batch_norm_epsilon
    if global_params.data_format == 'channels_first':
      self._channel_axis = 1
      self._spatial_dims = [2, 3]
    else:
      self._channel_axis = -1
      self._spatial_dims = [1, 2]
    self.has_se = (self._block_args.se_ratio is not None) and (
        self._block_args.se_ratio > 0) and (self._block_args.se_ratio <= 1)

    self.endpoints = None
    self.runtimes = layer_runtimes
    self.dropout_rate = dropout_rate

    self._search_space = global_params.search_space
    # Builds the block accordings to arguments.
    self._build()

  def _build(self):
    """Builds MBConv block according to the arguments."""
    filters = self._block_args.input_filters * self._block_args.expand_ratio
    if self._block_args.expand_ratio != 1:
      # Expansion phase:
      self._expand_conv = tf.keras.layers.Conv2D(
          filters,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=False)
      self._bn0 = tf.layers.BatchNormalization(
          axis=self._channel_axis,
          momentum=self._batch_norm_momentum,
          epsilon=self._batch_norm_epsilon,
          fused=True)

    kernel_size = self._block_args.kernel_size
    if self._search_space is None: #  for "default" layers

      # Default depth-wise convolution phase:
      self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
        [kernel_size, kernel_size],
        strides=self._block_args.strides,
        depthwise_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False)

    # Learnable Depth-wise convolution Superkernel
    elif self._search_space == 'mnasnet': 
      self._depthwise_conv = DepthwiseConv2DMasked(
        [kernel_size, kernel_size],
        strides=self._block_args.strides,
        depthwise_initializer=conv_kernel_initializer,
        padding='same', runtimes=self.runtimes,
        dropout_rate=self.dropout_rate,
        use_bias=False)

    else:
      raise NotImplementedError('DepthConv not defined for %s' % self._search_space)

    self._bn1 = tf.layers.BatchNormalization(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon,
        fused=True)

    if self.has_se:
      # why would you have SE in the supernet during search?
      assert 1 == 0 
      num_reduced_filters = max(
          1, int(self._block_args.input_filters * self._block_args.se_ratio))
      # Squeeze and Excitation layer.
      self._se_reduce = tf.keras.layers.Conv2D(
          num_reduced_filters,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=True)
      self._se_expand = tf.keras.layers.Conv2D(
          filters,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=True)

    # Output phase:
    filters = self._block_args.output_filters
    self._project_conv = tf.keras.layers.Conv2D(
        filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False)
    self._bn2 = tf.layers.BatchNormalization(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon,
        fused=True)

  def _call_se(self, input_tensor):
    """Call Squeeze and Excitation layer.

    Args:
      input_tensor: Tensor, a single input tensor for Squeeze/Excitation layer.

    Returns:
      A output tensor, which should have the same shape as input.
    """
    se_tensor = tf.reduce_mean(input_tensor, self._spatial_dims, keepdims=True)
    se_tensor = self._se_expand(tf.nn.relu(self._se_reduce(se_tensor)))
    tf.logging.info('Built Squeeze and Excitation with tensor shape: %s' %
                    (se_tensor.shape))
    return tf.sigmoid(se_tensor) * input_tensor

  def call(self, inputs, runtime, training=True):
    """Implementation of MBConvBlock call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.

    Returns:
      A output tensor.
    """
    tf.logging.info('Block input: %s shape: %s' % (inputs.name, inputs.shape))
    if self._block_args.expand_ratio != 1:
      x = tf.nn.relu(self._bn0(self._expand_conv(inputs), training=training))
    else:
      x = inputs
    tf.logging.info('Expand: %s shape: %s' % (x.name, x.shape))

    x, runtime = self._depthwise_conv(x, runtime)
    x = tf.nn.relu(self._bn1(x, training=training))
    tf.logging.info('DWConv: %s shape: %s' % (x.name, x.shape))

    if self.has_se:
      with tf.variable_scope('se'):
        x = self._call_se(x)

    self.endpoints = {'expansion_output': x}

    x = self._bn2(self._project_conv(x), training=training)
    if self._block_args.id_skip:
      if all(
          s == 1 for s in self._block_args.strides
      ) and self._block_args.input_filters == self._block_args.output_filters:
        x = tf.add(x, inputs)
    tf.logging.info('Project: %s shape: %s' % (x.name, x.shape))
    return x, runtime


class SinglePathSuperNet(tf.keras.Model):
  """class implements tf.keras.Model for SinglePath Supernet with superkernels
     More details: Fig.2 -- Single-Path NAS: https://arxiv.org/abs/(TBD)
     Based on MNasNet search space: https://arxiv.org/abs/1807.11626
  """

  def __init__(self, blocks_args=None, global_params=None,dropout_rate=None):
    """Initializes an `SuperNet` instance.

    Args:
      blocks_args: A list of BlockArgs to construct MBConv block modules.
      global_params: GlobalParams, a set of global parameters.

    Raises:
      ValueError: when blocks_args is not specified as a list.
    """
    super(SinglePathSuperNet, self).__init__()
    if not isinstance(blocks_args, list):
      raise ValueError('blocks_args should be a list.')
    self._global_params = global_params
    self._blocks_args = blocks_args
    self.endpoints = None
    self.dropout_rate = dropout_rate

    self._search_space = global_params.search_space

    tf.logging.info('Runtime model parsed')
    assert self._search_space == 'mnasnet' # currently supported one
    lutmodel_filename = "./pixel1_runtime_model.json"
    with open(lutmodel_filename, 'r') as f:
      self._runtime_lut = json.load(f)

    self._build()


  def _build(self):
    """Builds the supernet."""

    self._blocks = []
    # Builds blocks.
    for block_args in self._blocks_args:
      assert block_args.num_repeat > 0
      # Update block input and output filters based on depth multiplier.
      block_args = block_args._replace(
          input_filters=round_filters(block_args.input_filters,
                                      self._global_params),
          output_filters=round_filters(block_args.output_filters,
                                       self._global_params))

      # The first block needs to take care of stride and filter size increase.
      layer_runtimes = [self._runtime_lut[str(len(self._blocks))][str(i)] 
        for i in range(len(self._runtime_lut[str(len(self._blocks))].keys()))]
      self._blocks.append(MBConvBlock(block_args, self._global_params, 
                                    layer_runtimes, self.dropout_rate))
      if block_args.num_repeat > 1:
        # pylint: disable=protected-access
        block_args = block_args._replace(
            input_filters=block_args.output_filters, strides=[1, 1])
        # pylint: enable=protected-access
      for _ in xrange(block_args.num_repeat - 1):
        layer_runtimes = [self._runtime_lut[str(len(self._blocks))][str(i)] 
          for i in range(len(self._runtime_lut[str(len(self._blocks))].keys()))] + \
                [0.7] # neglibible (ms) value for skip-op (non-zero handling purposes)
        self._blocks.append(MBConvBlock(block_args, self._global_params, 
                                      layer_runtimes, self.dropout_rate))

    batch_norm_momentum = self._global_params.batch_norm_momentum
    batch_norm_epsilon = self._global_params.batch_norm_epsilon
    if self._global_params.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1

    # Stem part.
    self._conv_stem = tf.keras.layers.Conv2D(
        filters=round_filters(32, self._global_params),
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False)
    self._bn0 = tf.layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon,
        fused=True)

    # Head part.
    self._conv_head = tf.keras.layers.Conv2D(
        filters=1280,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False)
    self._bn1 = tf.layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon,
        fused=True)

    self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
        data_format=self._global_params.data_format)
    self._fc = tf.keras.layers.Dense(
        self._global_params.num_classes,
        kernel_initializer=dense_kernel_initializer)

    if self._global_params.dropout_rate > 0:
      self._dropout = tf.keras.layers.Dropout(self._global_params.dropout_rate)
    else:
      self._dropout = None

  def call(self, inputs, training=True):
    """Implementation of SuperNet call().

    Args:
      inputs: input tensors.
      training: boolean, whether the model is constructed for training.

    Returns:
      output tensors.
    """
    outputs = None
    self.endpoints = {}
    self.indicators = {}

    # rest of runtime (i.e., stem, head, logits, block0, block21)
    total_runtime = 19.5999

    # Calls Stem layers
    with tf.variable_scope('mnas_stem'):
      outputs = tf.nn.relu(
          self._bn0(self._conv_stem(inputs), training=training))
    tf.logging.info('Built stem layers with output shape: %s' % outputs.shape)
    self.endpoints['stem'] = outputs
    # Calls blocks.
    for idx, block in enumerate(self._blocks):
      with tf.variable_scope('mnas_blocks_%s' % idx):
        outputs, total_runtime = block.call(outputs, total_runtime, training=training)
        self.endpoints['block_%s' % idx] = outputs
        # the indicator decisions 
        if block._depthwise_conv.custom:
          self.indicators['block_%s' % idx] = {
                  'd5x5': block._depthwise_conv.d5x5,
                  'd50c': block._depthwise_conv.d50c,
                  'd100c': block._depthwise_conv.d100c}

        if block.endpoints:
          for k, v in six.iteritems(block.endpoints):
            self.endpoints['block_%s/%s' % (idx, k)] = v
    # Calls final layers and returns logits.
    with tf.variable_scope('mnas_head'):
      outputs = tf.nn.relu(
          self._bn1(self._conv_head(outputs), training=training))
      outputs = self._avg_pooling(outputs)
      if self._dropout:
        outputs = self._dropout(outputs, training=training)
      outputs = self._fc(outputs)
      self.endpoints['head'] = outputs

    return outputs, total_runtime
