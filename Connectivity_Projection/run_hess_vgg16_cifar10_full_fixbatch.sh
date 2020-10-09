#!/bin/sh -l

mkdir -p output/cifar10/hessian/

for MU in 0.51 0.6 0.7; do 
    for i in 10 20 30 40 ; do 
        python -u hess_diff.py --dir=./cifar10_save/endpoints/VGG16_fixbatch \
            --data_path=./cifar10_save/data/ \
            --model=VGG16 \
            --dataset=CIFAR10 \
            --use_test \
            --mu=$MU \
            --num_eigenthings=10 \
            --mode=lanczos \
            --batch_size=32 \
            --max_samples=32 \
            --full_dataset \
            --cuda \
            --resume_new='./cifar10_save/endpoints/VGG16_fixbatch/grda_c0.001_mu'$MU'/checkpoint-'$i'.pt' \
            --resume='./cifar10_save/endpoints/VGG16_fixbatch/sgd/checkpoint-'$i'.pt'  | tee -a  'output/cifar10/hessian/hess-cifar10-vgg16_fix_top10.txt' ; 
    done; 
done

for MU in 0.51 0.6 0.7; do 
    for i in `seq 50 50 600` ; do 
        python -u hess_diff.py --dir=./cifar10_save/endpoints/VGG16_fixbatch \
            --data_path=./cifar10_save/data/ \
            --model=VGG16 \
            --dataset=CIFAR10 \
            --use_test \
            --mu=$MU \
            --num_eigenthings=10 \
            --mode=lanczos \
            --batch_size=32 \
            --max_samples=32 \
            --full_dataset \
            --cuda \
            --resume_new='./cifar10_save/endpoints/VGG16_fixbatch/grda_c0.001_mu'$MU'/checkpoint-'$i'.pt' \
            --resume='./cifar10_save/endpoints/VGG16_fixbatch/sgd/checkpoint-'$i'.pt'  | tee -a  'output/cifar10/hessian/hess-cifar10-vgg16_fix_top10.txt' ; 
    done; 
done
