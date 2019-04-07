# MNasNet-based TFLite networks: Runtime profiling and modeling

## Profiling

To populate the runtime lookup-table (LUT) inference runtime model, 
generate .tflite models with different MBConv types (different 
kernel and expansion ratio values). The scripts below automate 
this process.

### Specific steps

1. To generate tflite models for all different MBConvs, run:

```
bash gen_tflite_models.sh
```
which executes
```
python main_tflite.py --tpu=$TPU_NAME --data_dir=$DATA_DIR --model_dir=${STORAGE_BUCKET}/model-runtime-model/model-tflite-$d-$k-$e --export_dir=$(pwd)/tflite-models/model-$d-$k-$e --depth_multiplier=$d --kernel=$k --expratio=$e --mode=train --post_quantize=True
```
which uses the '--export_dir' flag to generate the TFLite floal and quantized models,
following the MNasNet+TFLite [documentation](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet#serve-the-exported-model-in-tflite).



