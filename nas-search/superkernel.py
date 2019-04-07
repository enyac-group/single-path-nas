# authors: dstamoulis, ruizhoud
#
# Code extends codebase from the "MNasNet on TPU" GitHub repo:
# https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
#
# This project incorporates material from the project listed above, and it
# is accessible under their original license terms (Apache License 2.0)
# ==============================================================================
"""Contains the searchable superkernel definition based on the 
   Single-Path search space formulation.

[1] D. Stamoulis et al., Single-Path NAS: Designing Hardware-Efficient 
    ConvNets in less than 4 Hours. arXiv:(TBD)
"""


import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import InputSpec


def Indicator(x):
  #TypeError: Value passed to parameter 'x' has DataType bool not in list of allowed values: 
  #           bfloat16, float16, float32, float64, uint8, int8, uint16, int16, 
  #           int32, int64, complex64, complex128
  #return tf.stop_gradient((x>=0) - tf.sigmoid(x)) + tf.sigmoid(x)
  return tf.stop_gradient(tf.to_float(x>=0) - tf.sigmoid(x)) + tf.sigmoid(x)


def sample_gumbel(shape, eps=1e-20):
  U = tf.random_uniform(shape, minval=0, maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)


class DepthwiseConv2DMasked(tf.keras.layers.DepthwiseConv2D):
  def __init__(self, 
               kernel_size,
               strides,
               depthwise_initializer, 
               padding,
               use_bias,               
               runtimes=None,
               dropout_rate=None,
               **kwargs):
    
    super(DepthwiseConv2DMasked, self).__init__(
            kernel_size=kernel_size, 
            strides=strides, 
            depthwise_initializer=depthwise_initializer, 
            padding=padding, 
            use_bias=use_bias,
            **kwargs)

    self.runtimes = runtimes
    self.dropout_rate = tf.stop_gradient(dropout_rate)

    if kernel_size[0] != 5: # normal Depthwise type
      self.custom = False
    else:
      self.custom = True 
      if self.runtimes is not None:
        self.R50c = K.cast_to_floatx(self.runtimes[2]) # 50% of the 5x5
        self.R100c = K.cast_to_floatx(self.runtimes[3]) # 100% of the 5x5
        self.R5x5 = K.cast_to_floatx(self.runtimes[3]) # 5x5 for 100%
        self.R3x3 = K.cast_to_floatx(self.runtimes[1]) # 3x3 for 100%
      else:
        self.R50c = K.cast_to_floatx(0.0)
        self.R100c = K.cast_to_floatx(0.0)
        self.R5x5 = K.cast_to_floatx(0.0)
        self.R3x3 = K.cast_to_floatx(0.0)


  def build(self, input_shape):

    # NOTE: necessary for defining a Keras layer!
    # https://keras.io/layers/writing-your-own-keras-layers/
    # also need to build the superclass so that layer is populated w/ weights!
    super(DepthwiseConv2DMasked, self).build(input_shape)

    runtime = 0.0 

    if not self.custom:
      # back to typical DepthwiseConv2D
      self.depthwise_kernel_masked = self.depthwise_kernel
      self.runtime_reg = runtime
      
    else:

      # our implementation is channels_last
      assert self.data_format == 'channels_last'
      assert len(input_shape) == 4

      kernel_shape = self.depthwise_kernel.shape
      assert kernel_shape[-1] == 1 # I don't think we handle depth mult
      assert kernel_shape[0] == 5 # you cannot mask out if it is 3x3 already! 
      assert kernel_shape[1] == 5 # you cannot mask out if it is 3x3 already! 

      # Thresholds
      self.t5x5 = self.add_weight(shape=(1,),initializer='zeros',name="t5x5")
      self.t50c = self.add_weight(shape=(1,),initializer='zeros',name="t50c")
      self.t100c = self.add_weight(shape=(1,),initializer='zeros',name="t100c")

      # create masks based on kernel_shape
      center_3x3 = np.zeros(kernel_shape)
      center_3x3[1:4,1:4,:,:] = 1.0 # center 3x3
      self.mask3x3 = tf.convert_to_tensor(center_3x3, 
                        dtype=self.t5x5.dtype)

      center_5x5 = np.ones(kernel_shape) - center_3x3 # 5x5 - center 3x3
      self.mask5x5 = tf.convert_to_tensor(center_5x5, 
                        dtype=self.t5x5.dtype)

      num_channels = int(kernel_shape[2])
      c50  = int(round(1.0*num_channels/2.0)) #  50 %
      c100 = int(round(2.0*num_channels/2.0)) # 100 %

      mask_50c = np.zeros(kernel_shape)
      mask_50c[:,:,0:c50,:] = 1.0 # from 0% to 50% channels
      self.mask50c = tf.convert_to_tensor(mask_50c, 
                        dtype=self.t5x5.dtype)

      mask_100c = np.zeros(kernel_shape)
      mask_100c[:,:,c50:c100,:] = 1.0 # from 50% to 100% channels
      self.mask100c = tf.convert_to_tensor(mask_100c, 
                        dtype=self.t5x5.dtype)

      #--> make indicator results "accessible" as separate vars
      kernel_3x3 = self.depthwise_kernel * self.mask3x3
      kernel_5x5 = self.depthwise_kernel * self.mask5x5
      self.norm5x5 = tf.norm(kernel_5x5)

      x5x5 = self.norm5x5 - self.t5x5
      if self.dropout_rate is not None: # zero-out with drop_prob_ 
        self.d5x5 = tf.nn.dropout(Indicator(x5x5), self.dropout_rate)
      else:
        self.d5x5 = Indicator(x5x5)


      depthwise_kernel_masked_outside = \
            kernel_3x3 + kernel_5x5 * self.d5x5 

      kernel_50c = depthwise_kernel_masked_outside * self.mask50c
      kernel_100c = depthwise_kernel_masked_outside * self.mask100c
      self.norm50c = tf.norm(kernel_50c)
      self.norm100c = tf.norm(kernel_100c)


      x100c = self.norm100c - self.t100c
      if self.dropout_rate is not None: # noise to add
        self.d100c = tf.nn.dropout(Indicator(x100c), self.dropout_rate)
      else:
        self.d100c = Indicator(x100c) 


      if self.strides[0] == 1 and len(self.runtimes) == 5:
        x50c = self.norm50c - self.t50c
        if self.dropout_rate is not None: # noise to add
          self.d50c = tf.nn.dropout(Indicator(x50c), self.dropout_rate)
        else:
          self.d50c = Indicator(x50c) 
      else: # you cannot drop all layers!
        self.d50c = 1.0

      self.depthwise_kernel_masked = \
            self.d50c * (kernel_50c + self.d100c *kernel_100c)

      # runtime term
      if self.runtimes is not None:
        ratio = self.R3x3 / self.R5x5
        runtime_channels = self.d50c * (self.R50c + self.d100c * (self.R100c-self.R50c)) 
        runtime = runtime_channels * ratio + runtime_channels * (1-ratio) * self.d5x5

      self.runtime_reg = runtime


  def call(self, inputs, total_runtime, training=None):
    outputs = K.depthwise_conv2d(
        inputs,
        self.depthwise_kernel_masked,
        strides=self.strides,
        padding=self.padding,
        dilation_rate=self.dilation_rate,
        data_format=self.data_format)

    if self.use_bias:
      outputs = K.bias_add(
              outputs,
              self.bias,
              data_format=self.data_format)

    total_runtime = total_runtime + self.runtime_reg

    if self.activation is not None:
      return self.activation(outputs), total_runtime

    return outputs, total_runtime 

