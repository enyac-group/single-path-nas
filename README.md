# Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours

## Requirements
* Access to Cloud TPUs ([Official Cloud TPU Tutorial](https://cloud.google.com/tpu/docs/tutorials/mnasnet))
* Tensorflow 1.12
* Python 3.5+

## Updates
* 04/05/19: Single-Path NAS search code released.

## Contents

* NAS Search [nas-search](/nas-search/): Employ NAS search 
* Runtime Modeling [runtime-modeling](/runtime-modeling/): Generate ConvNets to profile
* Train ConvNet [train-final](/train-final/): Fully train found ConvNet on ImageNet



## Citation
Please cite the Single-Path paper ([link](https://arxiv.org/abs/1904.02877)) 
in your publications if this repo helps your research:

    @inproceedings{stamoulis2019singlepath,
      author = {Stamoulis, Dimitrios and Ding, Ruizhou and Wang, Di and Lymberopoulos, Dimitrios and Priyantha, Bodhi and Liu, Jie and Marculescu, Diana}
      booktitle = {arXiv preprint arXiv:1904.02877},
      title = {Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours},
      year = {2019}
    }

