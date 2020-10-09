#!/bin/sh -l

mkdir -p cifar100_save/endpoints/ cifar100_save/curves/ cifar100_save/data/ output/cifar100/

for seed in 1 ; do 
    MODEL=WideResNet28x10_$seed
    mkdir -p output/cifar100/$MODEL
    MODEL_=WideResNet28x10
    TRANSFORM=ResNet
    ENDPOINT_DIR='cifar100_save/endpoints/'$MODEL
    DATA_DIR='cifar100_save/data/'
    OUTPUT_DIR='output/cifar100/'$MODEL
    ENDPOINT_END=${ENDPOINT_DIR}/sgd
    DATASET=CIFAR100
    CURVE_DIR_PREFIX='cifar100_save/curves/'$MODEL
    CURVE=Bezier

    GRDA_C=0.001
    for GRDA_MU in 0.501 0.55 0.6 0.65; do
        ENDPOINT_START=${ENDPOINT_DIR}/grda_c${GRDA_C}_mu${GRDA_MU}
        CURVE_DIR=${CURVE_DIR_PREFIX}/sgd-grda_c${GRDA_C}_mu${GRDA_MU}
        
        # eval curve
        python -u eval_curve.py --dir=$CURVE_DIR   --data_path=$DATA_DIR --dataset=$DATASET --model=$MODEL_ --transform=$TRANSFORM  --use_test --batch_size 128 --wd=0 --curve=$CURVE --num_bends=3 --ckpt=$CURVE_DIR/checkpoint-200.pt | tee $OUTPUT_DIR/sgd-grda_c${GRDA_C}_mu${GRDA_MU}_eval_curve_ep200.txt

        # eval plane
        python -u plane.py --dir=$CURVE_DIR  --data_path=$DATA_DIR --dataset=$DATASET --model=$MODEL_ --transform=$TRANSFORM --use_test --batch_size 128 --grid_points=29  --curve=$CURVE --num_bends=3 --wd=0 --ckpt=$CURVE_DIR/checkpoint-200.pt | tee $OUTPUT_DIR/sgd-grda_c${GRDA_C}_mu${GRDA_MU}_eval_plane_ep200.txt

        # draw plane
        python plane_plot.py --dir=$CURVE_DIR 
    done;
done;
