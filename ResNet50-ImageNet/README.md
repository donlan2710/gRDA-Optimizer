# Code for gRDA on ResNet50-ImageNet

The code is for NeurIPS 2020 submission 'Directional Pruning of Deep Neural Networks'. 

More specifically, this folder contains code of running experiments on ResNet50-ImageNet.

## Installation

This code is mainly from the official PyTorch code example [ImageNet training in PyTorch](https://github.com/pytorch/examples/blob/234bcff4a2d8480f156799e6b9baae06f7ddc96a/imagenet/main.py). Please follow their manual if you have any question.

### Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)


## Using the code

Please use `chmod +x *.sh` to make the bash files executable. Or you can use `bash replace_with_bash_script_name.sh` to run the bash files we mention below.

For fixed learning rate, please run `./run_fixed_lr.sh`

For changing learning rate, we present two learning rate schedules:

1. The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs. Please run `./run_lr_schedule_30ep0.1.sh`
2. Another learning rate schedule starts at 0.1 and decays by a factor of 10 at 140 epochs. Please run `./run_lr_schedule_140ep0.1.sh`

The output training logs are saved in `./outputs/`. 
The model checkpoints are saved in `./save_models/`.

After training, we can calculate the layerwise sparsity by `./print_sparsity_per_layer.sh`.