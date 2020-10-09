#!/bin/sh -l

mkdir -p cifar10_save/endpoints/ cifar10_save/curves/ cifar10_save/data/ output/cifar10/

for seed in 1 2 3 4 5 6 7 8 9 10 ; do 
    MODEL=VGG16_$seed
    mkdir -p output/cifar10/$MODEL
    MODEL_=VGG16
    TRANSFORM=VGG
    ENDPOINT_DIR='cifar10_save/endpoints/'$MODEL
    DATA_DIR='cifar10_save/data/'
    OUTPUT_DIR='output/cifar10/'$MODEL
    ENDPOINT_END=${ENDPOINT_DIR}/sgd
    DATASET=CIFAR10

    # train end point
    python3 -u train.py --dir=$ENDPOINT_END     --data_path=$DATA_DIR --dataset=$DATASET --model=$MODEL_ --transform=$TRANSFORM --use_test --lr=0.1 --momentum 0 --wd=0 --batch_size=128 --epochs=600 | tee $OUTPUT_DIR/sgd.txt

    # train start point
    GRDA_C=0.0005
    for GRDA_MU in 0.4 0.51 0.6 0.7; do
        ENDPOINT_START=${ENDPOINT_DIR}/grda_c${GRDA_C}_mu${GRDA_MU}
        python3 -u train.py --dir=$ENDPOINT_START   --data_path=$DATA_DIR --dataset=$DATASET --model=$MODEL_ --transform=$TRANSFORM --use_test --lr=0.1 --momentum=0 --wd=0 --batch_size=128 --epochs=600 --grda=True --c=$GRDA_C --mu=$GRDA_MU --resume=$ENDPOINT_END/checkpoint-0.pt | tee $OUTPUT_DIR/grda_c${GRDA_C}_mu${GRDA_MU}.txt
    done;
done;