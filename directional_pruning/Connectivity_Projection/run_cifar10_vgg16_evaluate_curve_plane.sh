#!/bin/sh -l

mkdir -p cifar10_save/endpoints/ cifar10_save/curves/ cifar10_save/data/ output/cifar10/

for seed in 1 ; do 
    MODEL=VGG16_$seed
    mkdir -p output/cifar10/$MODEL
    MODEL_=VGG16
    TRANSFORM=VGG
    DATA_DIR='cifar10_save/data/'
    OUTPUT_DIR='output/cifar10/'$MODEL
    DATASET=CIFAR10
    CURVE_DIR_PREFIX='cifar10_save/curves/'$MODEL
    CURVE=Bezier

    GRDA_C=0.0005
    for GRDA_MU in 0.4 0.51 0.6 0.7; do
        CURVE_DIR=${CURVE_DIR_PREFIX}/sgd-grda_c${GRDA_C}_mu${GRDA_MU}
        
        # eval curve
        python -u eval_curve.py --dir=$CURVE_DIR   --data_path=$DATA_DIR --dataset=$DATASET --model=$MODEL_ --transform=$TRANSFORM  --use_test --batch_size 128 --wd=0 --curve=$CURVE --num_bends=3 --ckpt=$CURVE_DIR/checkpoint-300.pt | tee $OUTPUT_DIR/sgd-grda_c${GRDA_C}_mu${GRDA_MU}_eval_curve_ep300.txt

        # eval plane
        python -u plane.py --dir=$CURVE_DIR  --data_path=$DATA_DIR --dataset=$DATASET --model=$MODEL_ --transform=$TRANSFORM --use_test --batch_size 128 --grid_points=29  --curve=$CURVE --num_bends=3 --wd=0 --ckpt=$CURVE_DIR/checkpoint-300.pt | tee $OUTPUT_DIR/sgd-grda_c${GRDA_C}_mu${GRDA_MU}_eval_plane_ep300.txt

        # draw plane
        python plane_plot.py --dir=$CURVE_DIR 
    done;
done;
