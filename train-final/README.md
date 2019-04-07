# Train ConvNet on ImageNet

## Train ConvNet on ImageNet 

To train the found ConvNet (the result of [nas-search/](/nas-search/)), 
first the NAS-decisions (ConvNet encoding) are parsed from the 
output_dir of the NAS search (defined with the --parse_search_dir) flag. 
Then the parsed ConvNet arch is trained for 350 epochs on ImageNet. 
The training follows the MnasNet training schedule and hyper-parameters
from the original MnasNet-TPU repo.

### Specific steps

1. Setting up ImageNet dataset

To setup the ImageNet follow the instructions from [here](https://cloud.google.com/tpu/docs/tutorials/amoebanet#full-dataset)  


2. Setting up TPU-related ENV variables:
```
export STORAGE_BUCKET=gs://{your-bucket-name-here} 
export DATA_DIR=${STORAGE_BUCKET}/{imagenet-dataset-location}
export PARSE_DIR=${STORAGE_BUCKET}/model-single-path-search
export OUTPUT_DIR=${STORAGE_BUCKET}/model-single-path-train-final
export TPU_NAME=node-{your-tpu-name}
```

3. Launch training:
```
lambda_val=0.020; python main.py --tpu=$TPU_NAME --data_dir=$DATA_DIR --model_dir=${OUTPUT_DIR}/lambda-val-${lambda_val}/ --parse_search_dir=${PARSE_DIR}/lambda-val-${lambda_val}/

```

