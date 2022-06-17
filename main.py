import argparse
import os
import random
import warnings
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from dataset.utils import create_dataset
from common.utils import prepare_folders, save_checkpoint
from tensorboardX import SummaryWriter
from alg.utils import create_alg


parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--alg', default='standard', help='alg setting')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for dataset sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of dataset loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--gpu_str', default=None, type=str,
                    help='GPU id to use.')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--aux_set', type=str, default='TinyImages')


parser.add_argument('--ood_num', type=int, default=-1)
parser.add_argument('-ab', '--aux_batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lambda_o', type=float, default=0, help='lambda')
parser.add_argument('--alpha', type=float, default=-1, help='alpha')


best_acc1 = 0


def main():
    args = parser.parse_args()
    args.store_name = '_'.join(
        [args.dataset, args.alg, args.loss_type, args.imb_type, str(args.imb_factor), args.exp_str])
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable dataset parallelism.')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_str

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu


    cudnn.benchmark = True

    # Data loading code
    train_dataset, val_dataset, cls_num_list = create_dataset(args)
    args.cls_num_list = cls_num_list

    args.arch = "resnet32"

    num_classes = train_dataset.cls_num
    args.num_classes = num_classes

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    alg = create_alg(args, gpu, num_classes, cls_num_list, train_dataset)

    for epoch in range(args.start_epoch, args.epochs):
        # random.shuffle(alg.ood_data.labels)

        alg.adjust_learning_rate(epoch, args)
        # train for one epoch
        alg.train(train_loader, epoch, log_training, tf_writer)

        # evaluate on validation set
        acc1 = alg.validate(val_loader, epoch, log_testing, tf_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': alg.net.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': alg.optimizer.state_dict(),
        }, is_best)

    alg.validate(train_loader, args.epochs, log_testing, tf_writer, "Train")


if __name__ == '__main__':
    main()