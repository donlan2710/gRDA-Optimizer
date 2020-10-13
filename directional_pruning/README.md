# Code for gRDA

The code is for the paper **Directional Pruning of Deep Neural Networks** published in *NeurIPS 2020*. 

## Content 

For reproducing results: (the usage of the code is in each folder `./README.md`)

- Section 4.1: ResNet50 on the ImageNet. Please check `./ResNet50-ImageNet/`.
- Section 4.2: Connectivity between the minima of gRDA and SGD. Please check `./Connectivity_Projection/`
- Section 4.3: Direction of w^gRDA − w^SGD . Please check `./Connectivity_Projection/`

For reproducing figures from results: (we also include our results in the folders so you don't need to actually run the code again.)

- Figure 1: directional pruning in introduction. Please check `./Figures/intro/connection/`.
- Figure 2: ||w||_2/||w||_1. Please check `./Figures/intro/w2_over_w1/`. The code for the data is in `./Connectivity_Projection/`.
- Figure 3: ResNet50 on the ImageNet. Please check `./Figures/imagenet/training_log/`.
- Figure 4 (left): gRDA results compared to other results from Gale et al. Please check `./Figures/imagenet/gale_magv2/`.
- Figure 4 (right): layerwise sparsity. Please check `./Figures/imagenet/layerwise/`.
- Figure 5: connectivity between models trained by SGD and gRDA. Please check `./Figures/connectivity/`.
- Figure 6: fraction of weights trained by SGD and gRDA on the top10 eigenspace of Hessian. Please check `./Figures/projection/`.
- Figure 7: gRDA results compared to other results from Gale et al. Please check `./Figures/appendix/gale/`.
- Figure 8: learning trajectories for VGG16 on CIFAR-10. Please check `./Figures/appendix/train_cifar10/vgg16/`.
- Figure 9: learning trajectories for WRN28x10 on CIFAR-10. Please check `./Figures/appendix/train_cifar10/wrn28x10/`.
- Figure 10: learning trajectories for WRN28x10 on CIFAR-100. Please check `./Figures/appendix/train_cifar100/wrn28x10/`.
- Figure 11: connectivity between models trained by SGD and gRDA. Please check `./Figures/appendix/connectivity/`.
