
import argparse
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F

import curves
import data
import models
import utils
import numpy as np
import scipy
import scipy.stats

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='DNN curve training')
parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--log_path', type=str, default=None, metavar='PATH',
                    help='path to save log (default: None)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                    help='model name (default: None)')

parser.set_defaults(init_linear=True)
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--total_epoch', type=int, default=200, metavar='N',
                    help='number of epochs trained (default: 200)')



args = parser.parse_args()



loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    256, #args.batch_size,
    8, #args.num_workers,
    args.transform,
    True #args.use_test
)

architecture = getattr(models, args.model)

model = architecture.base(num_classes=num_classes, **architecture.kwargs)



tb_logger = SummaryWriter(args.log_path)
if not os.path.exists(args.log_path):
    print('Create {}.'.format(args.log_path))
    os.makedirs(args.log_path)
print("log file at:", args.log_path)

for epoch_idx in range(0, args.total_epoch):
    model_path = args.resume + str(epoch_idx) + ".pt"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state'])

    weight1 = torch.cat([torch.flatten(param).data for name, param in model.named_parameters() if param.requires_grad])
    weight1 = weight1.double() # to get rid of the limit of 16777216

    l1_norm = torch.norm(weight1, p=1).item()
    l2_norm = torch.norm(weight1, p=2).item()
    l0_norm = torch.norm(weight1, p=0).item()
    l_inf_norm = torch.norm(weight1, p=float("inf")).item()

    tb_logger.add_scalar('Train/L1', l1_norm, epoch_idx)
    tb_logger.add_scalar('Train/L2', l2_norm, epoch_idx)
    tb_logger.add_scalar('Train/L2_over_L1', l2_norm/l1_norm, epoch_idx)
    tb_logger.add_scalar('Train/L0', l0_norm, epoch_idx)
    tb_logger.add_scalar('Train/L_inf', l_inf_norm, epoch_idx)

    tb_logger.flush()

    print("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}".format(epoch_idx, l1_norm, l2_norm, l0_norm, l_inf_norm, model_path))

tb_logger.close()
