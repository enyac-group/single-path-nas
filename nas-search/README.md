# Single-Path NAS Search 

## Employ NAS Search 

To find a hardware-efficient ConvNet, launch a NAS search for different lambda 

### Specific steps

1. Setting up ImageNet dataset

To setup the ImageNet follow the instructions from [here](https://cloud.google.com/tpu/docs/tutorials/amoebanet#full-dataset)  

2. Setting up TPU-related ENV variables:
```
export STORAGE_BUCKET=gs://{your-bucket-name-here} 
export DATA_DIR=${STORAGE_BUCKET}/{imagenet-dataset-location}
export OUTPUT_DIR=${STORAGE_BUCKET}/model-single-path-search
export TPU_NAME=node-{your-tpu-name}
```
3. Launch NAS search:
```
lambda_val=0.020; python search_main.py --tpu=$TPU_NAME --data_dir=$DATA_DIR --model_dir=${OUTPUT_DIR}/lambda-val-${lambda_val}/ --runtime_lambda_val=${lambda_val} 
```

Output: To output the NAS-decision variables (indication functions values), the [search_main.py](/nas-search/search_main.py) uses host_call to send the indicator-values to the host CPU (alongside the loss, learning rate, etc. terms logged by default).

N.B. The TPU-repo default TensorBoard visualization 
([official link](https://cloud.google.com/tpu/docs/tensorboard-setup)) has limited functionality
(e.g., requires writing to a port and launcing the VM with the ctpu tool). We instead
directly parse the TF-events from the output folder (see next)


## Parsing final architectural decisions (final arch)

```
cd ./plot-progress/
lambda_val=0.020; python parse_search_output.py ${OUTPUT_DIR}/lambda-val-${lambda_val}/ netarch
cd ..
```

The script uses TF's EventAccumulator to parse the NAS-decision variables (indicator values); it prints the MBConv types of the ConvNet (following the MNasNet encoding)


## Visualizing NAS search progress


```
cd ./plot-progress/
lambda_val=0.020; python parse_search_output.py ${OUTPUT_DIR}/lambda-val-${lambda_val}/ progress
cd ..
```
The output of the script is the following:
![Runtime progress](/nas-search/plot-progress/spnas_progress_runtime.png)
![CE Loss progress](/nas-search/plot-progress/spnas_progress_ce.png)



