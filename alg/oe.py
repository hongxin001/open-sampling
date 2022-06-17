from alg.common.base_framework import BaseAlg
import torch.nn.functional as F
import torchvision.transforms as trn
from dataset.tinyimages_80mn_loader import TinyImages
import torch
import numpy as np
from dataset.utils import create_ood_dataset, create_ood_noise
from common.utils import AverageMeter, accuracy, ImbalancedDatasetSampler
import time
import abc, os

class OEScratch(BaseAlg):
    __metaclass__ = abc.ABCMeta

    def __init__(self, args, gpu, num_classes, cls_num_list, train_dataset):
        super(OEScratch, self).__init__(args, gpu, num_classes, cls_num_list, train_dataset)

        if args.aux_set not in ["Gaussian", "Rademacher", "Blob"]:
            self.ood_data = create_ood_dataset(args)
        else:
            self.ood_data = create_ood_noise(args.aux_set, args.ood_num, 1)
        import math
        # ood_batch_size = int(args.batch_size * (args.ood_num / (args.ood_num + len(train_loader.dataset))))
        # ood_batch_size = math.ceil(len(self.ood_data) / len(train_loader))
        self.train_loader_out = torch.utils.data.DataLoader(
            self.ood_data,
            batch_size=args.aux_batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    def train(self, train_loader, epoch, log, tf_writer):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # switch to train mode
        self.net.train()

        end = time.time()

        batch_num = min(len(train_loader), len(self.train_loader_out))
        for batch_idx, (in_set, out_set) in enumerate(zip(train_loader, self.train_loader_out)):
            # measure dataset loading time
            data_time.update(time.time() - end)

            loss, acc1, acc5 = self.train_batch_with_out(batch_idx, in_set, out_set, epoch)

            losses.update(loss.item(), in_set[0].shape[0])
            top1.update(acc1[0], in_set[0].shape[0])
            top5.update(acc5[0], in_set[0].shape[0])

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % self.args.print_freq == 0:
                output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, batch_idx, batch_num, batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5,
                    lr=self.optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
                print(output)
                log.write(output + '\n')
                log.flush()

        tf_writer.add_scalar('loss/train', losses.avg, epoch)
        tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
        tf_writer.add_scalar('lr', self.optimizer.param_groups[-1]['lr'], epoch)

    @abc.abstractmethod
    def train_batch_with_out(self, batch_idx, in_set, out_set, epoch):
        pass

