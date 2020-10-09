#!/bin/sh -l

mkdir -p cifar10_save/endpoints/ cifar10_save/curves/ cifar10_save/data/ output/cifar10/

for seed in 1 ; do 
    MODEL=VGG16_$seed
    mkdir -p output/cifar10/$MODEL
    MODEL_=VGG16
    TRANSFORM=VGG
    ENDPOINT_DIR='cifar10_save/endpoints/'$MODEL
    DATA_DIR='cifar10_save/data/'
    OUTPUT_DIR='output/cifar10/'$MODEL
    ENDPOINT_END=${ENDPOINT_DIR}/sgd
    DATASET=CIFAR10
    CURVE_DIR_PREFIX='cifar10_save/curves/'$MODEL
    CURVE=Bezier

    GRDA_C=0.0005
    for GRDA_MU in 0.4 0.51 0.6 0.7; do
        ENDPOINT_START=${ENDPOINT_DIR}/grda_c${GRDA_C}_mu${GRDA_MU}
        CURVE_DIR=${CURVE_DIR_PREFIX}/sgd-grda_c${GRDA_C}_mu${GRDA_MU}

        python -u train.py --dir=$CURVE_DIR --data_path=$DATA_DIR --dataset=$DATASET --model=$MODEL_ --transform=$TRANSFORM --use_test --lr=0.1 --momentum 0.9 --wd=0 --batch_size=128 --epochs=300 --curve=$CURVE --num_bends=3 --init_start=$ENDPOINT_START/checkpoint-600.pt --init_end=$ENDPOINT_END/checkpoint-600.pt --fix_start --fix_end | tee $OUTPUT_DIR/sgd-grda_c${GRDA_C}_mu${GRDA_MU}_train_curve.txt
    done;
done;