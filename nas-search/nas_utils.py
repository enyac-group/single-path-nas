# This code extends codebase from the "MNasNet on TPU" GitHub repo:
# https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
#
# This project incorporates material from the project listed above, and it
# is accessible under their original license terms (Apache License 2.0)
# ==============================================================================
"""SinglePath utilities (MNasNet learning rate schedule + SinglePath dropout)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def build_learning_rate(initial_lr,
                        global_step,
                        steps_per_epoch=None,
                        lr_decay_type='exponential',
                        decay_factor=0.97,
                        decay_epochs=2.4,
                        total_steps=None,
                        warmup_epochs=5):
  """Build learning rate."""
  if lr_decay_type == 'exponential':
    assert steps_per_epoch is not None
    decay_steps = steps_per_epoch * decay_epochs
    lr = tf.train.exponential_decay(
        initial_lr, global_step, decay_steps, decay_factor, staircase=True)
  elif lr_decay_type == 'cosine':
    assert total_steps is not None
    lr = 0.5 * initial_lr * (
        1 + tf.cos(np.pi * tf.cast(global_step, tf.float32) / total_steps))
  elif lr_decay_type == 'constant':
    lr = initial_lr
  else:
    assert False, 'Unknown lr_decay_type : %s' % lr_decay_type

  if warmup_epochs:
    tf.logging.info('Learning rate warmup_epochs: %d' % warmup_epochs)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    warmup_lr = (
        initial_lr * tf.cast(global_step, tf.float32) / tf.cast(
            warmup_steps, tf.float32))
    lr = tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)

  return lr


def build_dropout_rate(global_step, warmup_steps=2502):
  tf.logging.info('Dropout rate warmup steps: %d' % warmup_steps)
  warmup_dropout_rate = tf.cast(0.6, tf.float32)
  final_dropout_rate = tf.cast(1e2, tf.float32)
  dropout_rate = tf.cond(global_step < warmup_steps, lambda: warmup_dropout_rate, 
               lambda:final_dropout_rate)
  return dropout_rate


def build_runtime_lambda(global_step, warmup_steps=2502, final_lambda=1.0):
  tf.logging.info('Runtime lambda starts after steps: %d' % warmup_steps)
  warmup_lambda_ = tf.cast(0.0, tf.float32)
  final_lambda_ = tf.cast(final_lambda, tf.float32)
  runtime_lambda = tf.cond(global_step < warmup_steps, lambda: warmup_lambda_, 
               lambda:final_lambda_)
  return runtime_lambda


def build_optimizer(learning_rate,
                    optimizer_name='rmsprop',
                    decay=0.9,
                    epsilon=0.001,
                    momentum=0.9):
  """Build optimizer."""
  if optimizer_name == 'sgd':
    tf.logging.info('Using SGD optimizer')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  elif optimizer_name == 'momentum':
    tf.logging.info('Using Momentum optimizer')
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=momentum)
  elif optimizer_name == 'rmsprop':
    tf.logging.info('Using RMSProp optimizer')
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay, momentum,
                                          epsilon)
  else:
    tf.logging.fatal('Unknown optimizer:', optimizer_name)

  return optimizer
