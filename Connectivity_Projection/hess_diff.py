
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

from hessian_eigenthings import compute_hessian_eigenthings
from hessian_eigenthings import HVPOperator


parser = argparse.ArgumentParser(description='DNN curve training')
parser.add_argument('--dir', type=str, default='/tmp/curve/', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=0, metavar='N',
                    help='number of workers (default: 0)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                    help='model name (default: None)')

parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')
parser.add_argument('--init_start', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init start point (default: None)')
parser.add_argument('--fix_start', dest='fix_start', action='store_true',
                    help='fix start point (default: off)')
parser.add_argument('--init_end', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init end point (default: None)')
parser.add_argument('--fix_end', dest='fix_end', action='store_true',
                    help='fix end point (default: off)')
parser.set_defaults(init_linear=True)
parser.add_argument('--mu', type=float, default=0.7, metavar='MU',
                    help='gRDA mu (default: 0.7)')
parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                    help='turns off linear initialization of intermediate points (default: on)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--resume_new', type=str, default=None, metavar='CKPT1',
                    help='checkpoint1 to resume training from (default: None)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                    help='save frequency (default: 50)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument(
    "--num_eigenthings",
    default=5,
    type=int,
    help="number of eigenvals/vecs to compute",
)
parser.add_argument(
    "--eval_batch_size", default=16, type=int, help="test set batch size"
)
parser.add_argument(
    "--num_steps", default=50, type=int, help="number of power iter steps"
)
parser.add_argument("--max_samples", default=1024, type=int) #2048
parser.add_argument("--cuda", action="store_true", help="if true, use CUDA/GPUs")
parser.add_argument(
    "--full_dataset",
    action="store_true",
    help="if true,\
                    loop over all batches in set for each gradient step",
)
parser.add_argument("--fname", default="", type=str)
parser.add_argument("--mode", type=str, choices=["power_iter", "lanczos"])


args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test
)

architecture = getattr(models, args.model)

model = architecture.base(num_classes=num_classes, **architecture.kwargs)
model1 = architecture.base(num_classes=num_classes, **architecture.kwargs)

start_epoch = 1
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state'])
print("Let's use", torch.cuda.device_count(), "GPUs!")
model = torch.nn.DataParallel(model).cuda()

if args.resume_new is not None:
    print('Resume training from %s' % args.resume_new)
    checkpoint = torch.load(args.resume_new)
    model1.load_state_dict(checkpoint['model_state'])
model1.cuda()

print("args.full_dataset:", args.full_dataset)

criterion = torch.nn.CrossEntropyLoss()

num = args.num_eigenthings

size = int(sum(p.numel() for p in model.parameters()))

grad = torch.cat([g.contiguous().view(-1) for g in model.parameters()])
grad -= torch.cat([g.contiguous().view(-1) for g in model1.parameters()])

weight1 = torch.cat([g.contiguous().view(-1) for g in model.parameters()])
weight2 = torch.cat([g.contiguous().view(-1) for g in model1.parameters()])

import time

from os import path
previous_calculated_eigen_file = args.resume + '.top' + str(num) + 'eigen' + ".npz"
if path.exists(previous_calculated_eigen_file):
    print("find previous calculated file at " + previous_calculated_eigen_file)
    time_start = time.time()
    pre_calculated_eigen_info = np.load(previous_calculated_eigen_file)
    tmp_eigen = pre_calculated_eigen_info['tmp_eigen']
    

    result = []
    for i in range(0, num):
        result.append(torch.pow(torch.norm(torch.matmul(torch.Tensor(tmp_eigen[i,]).cuda(),grad)),2).item())
        result.append(torch.pow(torch.norm(torch.matmul(torch.Tensor(tmp_eigen[i,]).cuda(),weight1)),2).item())
        result.append(torch.pow(torch.norm(torch.matmul(torch.Tensor(tmp_eigen[i,]).cuda(),weight2)),2).item())
        result.append(torch.pow(torch.norm(grad ),2).item())
        result.append(torch.pow(torch.norm(weight1 ),2).item())
        result.append(torch.pow(torch.norm(weight2 ),2).item())
    with open(os.path.join(args.dir, 'mu_'+str(args.mu)+'-projection_diff_new.txt'), 'a') as f:
        print("hello-e2", os.path.join(args.dir, 'mu_'+str(args.mu)+'-projection_diff_new.txt'))
        f.write(args.resume+'\t'+" ".join(map(str,  pre_calculated_eigen_info['tmp']))+"\t")
        f.write( ' '.join([ str(j) for j in result ]) )
        f.write('\n')

else:
    while True:
        time_start = time.time()
        eigenvals, eigenvecs = compute_hessian_eigenthings(
            model,
            loaders['train'],
            criterion,
            num*3,
            mode=args.mode,
            # power_iter_steps=args.num_steps,
            max_samples=args.max_samples,
            # momentum=args.momentum,
            full_dataset=args.full_dataset,
            use_gpu=args.cuda,
        )
        last = -np.array(range(1,(num+1)))
        tmp = eigenvals[last]
        tmp_eigen = eigenvecs[last,]

        np.savez(previous_calculated_eigen_file, tmp=tmp, tmp_eigen=tmp_eigen)

        if tmp[tmp<0].size == 0:
            result = []
            for i in range(0, num):
                result.append(torch.pow(torch.norm(torch.matmul(torch.Tensor(tmp_eigen[i,]).cuda(),grad)),2).item())
                result.append(torch.pow(torch.norm(torch.matmul(torch.Tensor(tmp_eigen[i,]).cuda(),weight1)),2).item())
                result.append(torch.pow(torch.norm(torch.matmul(torch.Tensor(tmp_eigen[i,]).cuda(),weight2)),2).item())
                result.append(torch.pow(torch.norm(grad ),2).item())
                result.append(torch.pow(torch.norm(weight1 ),2).item())
                result.append(torch.pow(torch.norm(weight2 ),2).item())
            with open(os.path.join(args.dir, 'mu_'+str(args.mu)+'-projection_diff_new.txt'), 'a') as f:
                f.write(args.resume+'\t'+" ".join(map(str, tmp))+"\t")
                f.write( ' '.join([ str(j) for j in result ]) )
                f.write('\n')
            break
        else:
            num = num+tmp[tmp<0].size*2

