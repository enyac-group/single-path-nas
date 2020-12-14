# author: dstamoulis
#
# This code extends codebase from the "MNasNet on TPU" GitHub repo:
# https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
#
# This project incorporates material from the project listed above, and it
# is accessible under their original license terms (Apache License 2.0)
# ==============================================================================
"""Creating .json FAI-PEP profiler-compatible file."""

def profiler_template(path):
 
  dirs = path.split('/')
  model_ = "mnasnet-backbone.tflite"
  name_ = dirs[-2]
  template = {"model": {
      "category": "CNN", 
      "cooldown": 30, 
      "description": "MNasNet model on TFLite", 
      "files": {
        "graph": {
          "filename": model_,
          "location": "/home/profiler/FAI-PEP/specifications/models/tflite/ext-search-space/"+name_+"/"+model_ 
        }
      }, 
      "format": "tflite", 
      "name": name_
    }, 
    "tests": [
      {
        "commands": [
          "{program} --graph={files.graph} --warmup_runs={warmup} --num_runs={iter} --input_layer=truediv --input_layer_shape=\"1,224,224,3\" --num_threads=1"
        ], 
        "identifier": name_ + "-6-thread", 
        "iter": 50, 
        "metric": "delay", 
        "warmup": 1,
        "platform_args": {
          "taskset": "8"
        }
      }
    ]
  }
  return template

import os, sys, json
if __name__ == '__main__':

  path = sys.argv[1]
  template = profiler_template(path)

  profiler_cfg_ = path + "profiler.json"
  if not os.path.exists(os.path.dirname(profiler_cfg_)):
      os.makedirs(os.path.dirname(profiler_cfg_))
  with open(profiler_cfg_, 'w') as f:
    json.dump(template, f)


