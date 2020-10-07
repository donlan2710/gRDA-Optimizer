#!/bin/sh -l

mkdir -p outputs/30ep0.1/ save_models/30ep0.1/

c=0.005
for mu in 0.85 0.8 0.75 0.7 0.65 ; do 
    python -u main_gRDA_print_sparse_save_model.py imagenet_data/ --resume save_models/fixed0.1/checkpoint_SGD_lr0.1_I.pth.tar -c $c --mu $mu -a resnet50 --save save_models/30ep0.1/ | tee outputs/30ep0.1/c${c}_mu${mu}.txt
done;

c=0.004
for mu in 0.75 0.7 0.65 ; do 
    python -u main_gRDA_print_sparse_save_model.py imagenet_data/ --resume save_models/fixed0.1/checkpoint_SGD_lr0.1_I.pth.tar -c $c --mu $mu -a resnet50 --save save_models/30ep0.1/ | tee outputs/30ep0.1/c${c}_mu${mu}.txt
done;

c=0.003
for mu in 0.75 0.7 ; do 
    python -u main_gRDA_print_sparse_save_model.py imagenet_data/ --resume save_models/fixed0.1/checkpoint_SGD_lr0.1_I.pth.tar -c $c --mu $mu -a resnet50 --save save_models/30ep0.1/ | tee outputs/30ep0.1/c${c}_mu${mu}.txt
done;

c=0.002
for mu in 0.75 ; do 
    python -u main_gRDA_print_sparse_save_model.py imagenet_data/ --resume save_models/fixed0.1/checkpoint_SGD_lr0.1_I.pth.tar -c $c --mu $mu -a resnet50 --save save_models/30ep0.1/ | tee outputs/30ep0.1/c${c}_mu${mu}.txt
done;

