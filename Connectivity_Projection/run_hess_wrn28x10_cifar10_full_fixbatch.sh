#!/bin/sh -l

mkdir -p output/cifar10/hessian/

for MU in 0.501 0.55 0.65; do 
    for i in 2 4 6 8; do 
        python -u hess_diff.py --dir=./cifar10_save/endpoints/WideResNet28x10_fixbatch \
            --data_path=./cifar10_save/data/ \
            --model=WideResNet28x10 \
            --dataset=CIFAR10 \
            --use_test \
            --mu=$MU \
            --num_eigenthings=10 \
            --mode=lanczos \
            --batch_size=32 \
            --max_samples=32 \
            --full_dataset \
            --cuda \
            --resume_new='./cifar10_save/endpoints/WideResNet28x10_fixbatch/grda_c0.001_mu'$MU'/checkpoint-'$i'.pt' \
            --resume='./cifar10_save/endpoints/WideResNet28x10_fixbatch/sgd/checkpoint-'$i'.pt'  | tee -a  'output/cifar10/hessian/hess-cifar10-wrn28x10_fix_top10.txt' ; 
    done; 
done

for MU in 0.501 0.55 0.65; do 
    for i in `seq 10 10 200` ; do 
        python -u hess_diff.py --dir=./cifar10_save/endpoints/WideResNet28x10_fixbatch \
            --data_path=./cifar10_save/data/ \
            --model=WideResNet28x10 \
            --dataset=CIFAR10 \
            --use_test \
            --mu=$MU \
            --num_eigenthings=10 \
            --mode=lanczos \
            --batch_size=32 \
            --max_samples=32 \
            --full_dataset \
            --cuda \
            --resume_new='./cifar10_save/endpoints/WideResNet28x10_fixbatch/grda_c0.001_mu'$MU'/checkpoint-'$i'.pt' \
            --resume='./cifar10_save/endpoints/WideResNet28x10_fixbatch/sgd/checkpoint-'$i'.pt'  | tee -a  'output/cifar10/hessian/hess-cifar10-wrn28x10_fix_top10.txt' ; 
    done; 
done
