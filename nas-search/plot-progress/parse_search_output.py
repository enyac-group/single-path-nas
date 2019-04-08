# Single-Path NAS (Apache License 2.0)
# ==============================================================================
"""Parsing the NAS search progress."""

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import OrderedDict

def parse_indicators_single_path_nas(path, tf_size_guidance):

  event_acc = EventAccumulator(path, tf_size_guidance)
  event_acc.Reload()

  # Show all tags in the log file
  tags = event_acc.Tags()['scalars']
  labels = ['t5x5_','t50c_','t100c_']
  inds = []
  for idx in range(20):
    layer_row = []
    for label_ in labels:
      summary_label_ = label_ + str(idx+1) 
      decision_ij = event_acc.Scalars(summary_label_)
      layer_row.append(decision_ij[-1].value)
    inds.append(layer_row)
  return inds


def encode_single_path_nas_arch(inds, hard=False):

  print('Sampling network')
  network = []
  candidate_ops = ['3x3-3', '3x3-6', '5x5-3', '5x5-6', 'skip']
  for layer_cnt in range(20):

    inds_row = inds[layer_cnt]
    print(inds_row)
    if inds_row == [0.0, 0.0, 0.0]:
      idx = 4 # skip 
    elif inds_row == [0.0, 0.0, 1.0]:
      idx = 4 # skip
    elif inds_row == [0.0, 1.0, 0.0]:
      idx = 0 # 3x3-3
    elif inds_row == [0.0, 1.0, 1.0]:
      idx = 1 # 3x3-6
    elif inds_row == [1.0, 0.0, 0.0]:
      idx = 4 # skip
    elif inds_row == [1.0, 0.0, 1.0]:
      idx = 4  # skip
    elif inds_row == [1.0, 1.0, 0.0]:
      idx = 2 # 5x5-3
    elif inds_row == [1.0, 1.0, 1.0]:
      idx = 3 # 5x5-6
    else:
      assert 0 == 1 # will crash
    network.append(candidate_ops[idx])
  return network


def parse_runtime(path, tf_size_guidance):

  event_acc = EventAccumulator(path, tf_size_guidance)
  event_acc.Reload()
  # Show all tags in the log file
  tags = event_acc.Tags()['scalars']
  al = event_acc.Scalars('alpha_1_2')
  runtime_row = []
  for i in range(len(al)):
    runtime_row.append(al[i].value)
  print(runtime_row)


def print_net(network):
  for idx, layer in enumerate(network):
    print(idx, layer)

def print_encoded_net(blocks_args):
  for idx, layer in enumerate(blocks_args):
    print(idx, layer)


def convnet_encoder(network):
  # this encodes our layer types to the mnasnet-based
  # encoding for the model generation!
  ichannels_ = ['_i16','_i24','_i40','_i80','_i96']
  inner_channels_ = ['_i24','_i40','_i80','_i96','_i192']
  ochannels_ = ['_o24','_o40','_o80','_o96','_o192']
  stride2_layers =  [0,4,8,12,16] # these you cannot drop

  block_cnt = 0
  # first bottleneck
  blocks_args = ['r1_k3_s11_e1_i32_o16_noskip']
  for stage_idx in range(5): # 5 groups of up to 4 layers
    for inner_block in range(4):
      layer_type = network[block_cnt]
      if layer_type == 'skip':
        assert block_cnt not in stride2_layers
      else:
        if layer_type == '3x3-3':
          kernel_sample, exp_ratio_sample = 'k3', 'e3'
        elif layer_type == '3x3-6':
          kernel_sample, exp_ratio_sample = 'k3', 'e3'
        elif layer_type == '5x5-3':
          kernel_sample, exp_ratio_sample = 'k5', 'e6'
        elif layer_type == '5x5-6':
          kernel_sample, exp_ratio_sample = 'k5', 'e6'

        # bug found! 1st block of 4th group does not drop!
        if block_cnt in stride2_layers and block_cnt != 12:
          stride_sample = '_s22_'
        else:
          stride_sample = '_s11_'

        if inner_block == 0:
          ich_ = ichannels_[stage_idx]
        else:
          ich_ = inner_channels_[stage_idx]

        next_block_encoding = 'r1_' + kernel_sample + \
            stride_sample + exp_ratio_sample + \
            ich_ + ochannels_[stage_idx]
        blocks_args.append(next_block_encoding)
      block_cnt += 1

  # last bottleneck
  blocks_args.append('r1_k3_s11_e6_i192_o320_noskip')
  return blocks_args


def parse_progress(path, tf_size_guidance):

  event_acc = EventAccumulator(path, tf_size_guidance)
  event_acc.Reload()
 
  tags = event_acc.Tags()['scalars']
  print(tags)

  # Show all tags in the log file
  tags = event_acc.Tags()['scalars']
  runtimes_scalar = event_acc.Scalars('runtime_ms')
  runtimes = [runtimes_scalar[i].value for i in range(len(runtimes_scalar))]

  loss_scalar = event_acc.Scalars('loss')
  loss = [loss_scalar[i].value for i in range(len(loss_scalar))]
  assert len(runtimes) == len(loss)

  return runtimes, loss


def plot_progress(runtimes, ce_loss):
  try:
    import matplotlib.pyplot as plt
  except:
    print("Matplotlib not installed. Try again...")
    print("Runtime", runtimes)  
    print("Loss", ce_loss)
    return False

  # Tensorboard-like smoothed_lie
  def tensorboard_smooth_line(list_, weight=0.95):
    list_sm = [list_[0]]
    for i, num in enumerate(list_[1:]):
      smoothed = list_sm[-1] * weight + (1 - weight) * num
      list_sm.append(smoothed)
      # list_sm.append(list_[-1])
    return list_sm

  steps_num = 10008
  smoothed_runtimes = tensorboard_smooth_line(runtimes, 0.8)
  smoothed_ce = tensorboard_smooth_line(ce_loss, 0.85)

  per_step = float(steps_num) / float(len(runtimes))
  steps = [per_step*(i+1) for i in range(len(runtimes))]

  plt.figure(figsize=(6.4,4.0))
  plt.plot(steps, runtimes, '-g', alpha=0.3)
  plt.plot(steps, smoothed_runtimes, '-g', linewidth=2)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.xlabel("Training Steps (NAS Search)", fontsize=16)
  plt.ylabel("Runtime term R", fontsize=16)
  plt.tight_layout()
  plt.savefig("progress_runtime.png")
  plt.close()

  plt.figure(figsize=(6.4,4.0))
  plt.plot(steps, ce_loss, '-b', alpha=0.3)
  plt.plot(steps, smoothed_ce, '-b')
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.xlabel("Training Steps (NAS Search)", fontsize=16)
  plt.ylabel("Cross Entropy CE term", fontsize=16)
  plt.tight_layout()
  plt.savefig("progress_ce.png")
  plt.close()

  return True

import os, sys
if __name__ == '__main__':

  if len(sys.argv) != 3:
    print("Argument: {Bucket-path} {log-action}")
    exit()
  
  log_file = sys.argv[1] 
  # Loading too much data is slow...
  tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
  }

  log_type = sys.argv[2]

  if log_type == 'progress':
    runtimes, ce_loss = parse_progress(log_file, tf_size_guidance)
    _ = plot_progress(runtimes, ce_loss)
  elif log_type == 'netarch':
    indicator_values = parse_indicators_single_path_nas(log_file, tf_size_guidance)
    network = encode_single_path_nas_arch(indicator_values)
    print("Net decisions (MBConv) per layer")
    print_net(network)
    print("MnasNet-like TPU compatible encoding")
    block_args = convnet_encoder(network)
    print_encoded_net(block_args)
  else:
    print("Not supported log action")


