#!/bin/sh -l

mkdir -p outputs/fixed0.1/ save_models/fixed0.1/

python -u main_sgd_save_model.py imagenet_data/ -a resnet50 --save save_models/fixed0.1/ | tee outputs/fixed0.1/sgd.txt

c=0.01
mu=0.4
python -u main_gRDA_print_sparse_save_model.py imagenet_data/ --resume save_models/fixed0.1/checkpoint_SGD_lr0.1_I.pth.tar -c $c --mu $mu -a resnet50 --save save_models/fixed0.1/ | tee outputs/fixed0.1/c${c}_mu${mu}.txt

c=0.005
for mu in 0.501 0.51 0.55 ; do 
    python -u main_gRDA_print_sparse_save_model.py imagenet_data/ --resume save_models/fixed0.1/checkpoint_SGD_lr0.1_I.pth.tar -c $c --mu $mu -a resnet50 --save save_models/fixed0.1/ | tee outputs/fixed0.1/c${c}_mu${mu}.txt
done;