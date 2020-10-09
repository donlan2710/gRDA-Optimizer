# Code for gRDA on Connectivity and Projection

The code is for NeurIPS 2020 submission 'Directional Pruning of Deep Neural Networks'. 

More specifically, this folder contains code of running experiments on **Connectivity** between the minima of gRDA and SGD and **Direction of w^gRDA − w^SGD (Projected in the subspace of top10 eigenvectors)**. We examine three settings: VGG16/CIFAR-10, WRN28x10/CIFAR-10, and WRN28x10/CIFAR-100.

## Installation

The connectivity part of code is mainly from the code of [Mode Connectivity and Fast Geometric Ensembling](https://github.com/timgaripov/dnn-mode-connectivity). The projection part of code is mainly from the code of [pytorch-hessian-eigenthings](https://github.com/noahgolmant/pytorch-hessian-eigenthings). Please follow their manuals if you have any question.

### Requirements

- `pip install -r requirements.txt`

## Using the code

Please use `chmod +x *.sh` to make the bash files executable. Or you can use `bash replace_with_bash_script_name.sh` to run the bash files we mention below.

### Training 

For VGG16/CIFAR-10, please run `./run_cifar10_vgg16_train.sh`

For WRN28x10/CIFAR-10, please run `./run_cifar10_wrn28x10_train.sh`

For WRN28x10/CIFAR-100, please run `./run_cifar100_wrn28x10_train.sh`

The output training logs are saved in the folder `./output/`.
The model checkpoints are saved in `./cifar10_save/endpoints/` and `./cifar100_save/endpoints/`.

### Figure 2: ||w||_2 / ||w||_1

Please run `./run_w2_over_w1.sh`. 

Then you can use [TensorBoard for PyTorch](https://github.com/lanpa/tensorboardX) to view the results by `tensorboard --logdir ./w2_over_l1/log/`.

### Connectivity

#### Training curves

For VGG16/CIFAR-10, please run `./run_cifar10_vgg16_train_curve.sh`

For WRN28x10/CIFAR-100, please run `./run_cifar100_wrn28x10_train_curve.sh`

The output training logs are saved in the folder `./output/`.
The model checkpoints are saved in `./cifar10_save/curves/` and `./cifar100_save/curves/`.

#### Evaluate curves and planes

The plane is a 2D plane containing the curve we have trained.

For VGG16/CIFAR-10, please run `./run_cifar10_vgg16_evaluate_curve_plane.sh`

For WRN28x10/CIFAR-100, please run `./run_cifar100_wrn28x10_evaluate_curve_plane.sh`

The output training logs are saved in the folder `./output/`.
The files of numpy data (for both curve and plane) are saved in `./cifar10_save/curves/` and `./cifar100_save/curves/`.

### Projection

Here we need to train the model again but with fixed minibatch to remove the uncertainty from data feeding (i.e. different orders of minibatches used in training with gRDA and SGD).

For VGG16/CIFAR-10, please run `./run_cifar10_vgg16_train_fixbatch.sh` and `./run_hess_vgg16_cifar10_full_fixbatch.sh`.

For WRN28x10/CIFAR-10, please run `./run_cifar10_wrn28x10_train_fixbatch.sh` and `run_hess_wrn28x10_cifar10_full_fixbatch.sh`.

The model checkpoints and the results are saved in the folder `./cifar10_save/endpoints/*_fixbatch/`.

