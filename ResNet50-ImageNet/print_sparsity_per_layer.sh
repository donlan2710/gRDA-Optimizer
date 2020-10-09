#!/bin/sh -l

mkdir -p outputs 

python -u pytorch_model_load_evaluate.py --resume save_models/140ep0.1/model_best_gRDA_lr0.01_c0.005_mu0.55_ep150.pth.tar > outputs/sparsity_gRDA_c0.005_mu0.55.csv