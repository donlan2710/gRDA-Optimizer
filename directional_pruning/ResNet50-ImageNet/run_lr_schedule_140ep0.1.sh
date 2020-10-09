#!/bin/sh -l

mkdir -p outputs/140ep0.1/ save_models/140ep0.1/

c=0.005
for mu in 0.7 0.65 0.6 0.55 0.51 0.501; do 
    python -u main_gRDA_print_sparse_save_model.py imagenet_data/ --resume save_models/fixed0.1/checkpoint_SGD_lr0.1_I.pth.tar -c $c --mu $mu -a resnet50 --save save_models/140ep0.1/ | tee outputs/140ep0.1/c${c}_mu${mu}.txt
done;

c=0.01
for mu in 0.4; do 
    python -u main_gRDA_print_sparse_save_model.py imagenet_data/ --resume save_models/fixed0.1/checkpoint_SGD_lr0.1_I.pth.tar -c $c --mu $mu -a resnet50 --save save_models/140ep0.1/ | tee outputs/140ep0.1/c${c}_mu${mu}.txt
done;