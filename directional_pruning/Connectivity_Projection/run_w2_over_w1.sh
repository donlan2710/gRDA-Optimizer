#!/bin/sh -l

mkdir -p w2_over_l1/log

for SEED in 1 2; do
    folder='./cifar10_save/endpoints/VGG16_'$SEED'/sgd'
    python -u w2_over_w1.py \
        --data_path=./cifar10_save/data/ \
        --log_path='./w2_over_l1/log/CIFAR10-VGG16-'$SEED'/sgd/' \
        --model=VGG16 \
        --dataset=CIFAR10 \
        --transform=VGG \
        --total_epoch=600 \
        --resume=$folder'/checkpoint-'  | tee $folder/l1_over_l1.txt   ; 

    for MU in 0.51 ; do 
        folder='./cifar10_save/endpoints/VGG16_'$SEED'/grda_c0.0005_mu'$MU
        python -u w2_over_w1.py \
            --data_path=./cifar10_save/data/ \
            --log_path='./w2_over_l1/log/CIFAR10-VGG16-'$SEED'/grda_c0.001_mu'$MU'/' \
            --model=VGG16 \
            --dataset=CIFAR10 \
            --transform=VGG \
            --total_epoch=600 \
            --resume=$folder'/checkpoint-'  | tee $folder/l1_over_l1.txt   ; 
    done;
done;

for SEED in 1 2; do
    folder='./cifar100_save/endpoints/WideResNet28x10_'$SEED'/sgd'
    python -u w2_over_w1.py \
        --data_path=./cifar100_save/data/ \
        --log_path='./w2_over_l1/log/CIFAR100-WideResNet28x10-'$SEED'/sgd/' \
        --model=WideResNet28x10 \
        --dataset=CIFAR100 \
        --transform=ResNet \
        --total_epoch=200 \
        --resume=$folder'/checkpoint-'  | tee $folder/l1_over_l1.txt   ; 

    for MU in 0.501 ; do 
        folder='./cifar100_save/endpoints/WideResNet28x10_'$SEED'/grda_c0.001_mu'$MU
        python -u w2_over_w1.py \
            --data_path=./cifar100_save/data/ \
            --log_path='./w2_over_l1/log/CIFAR100-WideResNet28x10-'$SEED'/grda_c0.001_mu'$MU'/' \
            --model=WideResNet28x10 \
            --dataset=CIFAR100 \
            --transform=ResNet \
            --total_epoch=200 \
            --resume=$folder'/checkpoint-'  | tee $folder/l1_over_l1.txt   ; 
    done;
done;
