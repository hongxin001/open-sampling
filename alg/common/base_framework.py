import time
from sklearn.metrics import confusion_matrix
from common.utils import AverageMeter, accuracy, ImbalancedDatasetSampler
import torch
import numpy as np
from model.utils import create_model
import abc, os
import warnings
import torch.nn as nn


class BaseAlg():
    __metaclass__ = abc.ABCMeta
    def __init__(self, args, gpu, num_classes, cls_num_list, train_dataset):
        self.args = args
        self.num_classes = num_classes
        self.gpu = gpu
        self.train_dataset = train_dataset

        if args.gpu is not None:
            self.device = torch.device('cuda:{}'.format(int(args.gpu)))
        else:
            self.device = torch.device('cuda')

        # create model
        print("=> creating model '{}'".format(args.arch))
        self.use_norm = True if args.loss_type == 'LDAM' else False

        self.net = create_model(num_classes, gpu, self.use_norm)

        self.optimizer = torch.optim.SGD(self.net.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        self.criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location='cuda:0')
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                if args.gpu is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = best_acc1.to(args.gpu)
                self.net.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))


    def train(self, train_loader, epoch, log, tf_writer):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # switch to train mode
        self.net.train()

        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            # measure dataset loading time
            data_time.update(time.time() - end)

            loss, acc1, acc5 = self.train_batch(i, input, target, epoch)

            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5,
                    lr=self.optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
                print(output)
                log.write(output + '\n')
                log.flush()


        tf_writer.add_scalar('loss/train', losses.avg, epoch)
        tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
        tf_writer.add_scalar('lr', self.optimizer.param_groups[-1]['lr'], epoch)

    def validate(self, val_loader, epoch, log=None, tf_writer=None, flag='val'):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # switch to evaluate mode
        self.net.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(val_loader):
                if self.gpu is not None:
                    input = input.cuda(self.gpu, non_blocking=True)
                target = target.cuda(self.gpu, non_blocking=True)

                # compute output
                output = self.net(input)
                loss = self.criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                if i % self.args.print_freq == 0:
                    output = ('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))
                    print(output)
            cf = confusion_matrix(all_targets, all_preds).astype(float)
            cls_cnt = cf.sum(axis=1)
            cls_hit = np.diag(cf)
            cls_acc = cls_hit / cls_cnt
            output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                      .format(flag=flag, top1=top1, top5=top5, loss=losses))
            print(output)

            if self.num_classes <=100:
                out_cls_acc = '%s Class Accuracy: %s' % (
                    flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
                print(out_cls_acc)
            if log is not None:
                log.write(output + '\n')
                if self.num_classes <=100:
                    log.write(out_cls_acc + '\n')
                log.flush()

            if tf_writer:
                tf_writer.add_scalar('loss/test_' + flag, losses.avg, epoch)
                tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
                tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
                if self.num_classes <= 100:
                    tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i): x for i, x in enumerate(cls_acc)}, epoch)

        return top1.avg



    def adjust_learning_rate(self, epoch, args):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        epoch = epoch + 1
        if epoch <= 5:
            lr = args.lr * epoch / 5
        elif epoch > 180:
            lr = args.lr * 0.0001
        elif epoch > 160:
            lr = args.lr * 0.01
        else:
            lr = args.lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr




    @abc.abstractmethod
    def train_batch(self, batch_idx, input, target, epoch):
        pass

    def resume_pretrain(self, dir):
        # optionally resume from a checkpoint
        if os.path.isfile(self.args.resume):
            print("=> loading checkpoint '{}'".format(self.args.resume))
            checkpoint = torch.load(self.args.resume, map_location='cuda:0')
            self.net.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(self.args.resume))



