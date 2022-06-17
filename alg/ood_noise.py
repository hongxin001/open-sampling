from alg.common.base_framework import BaseAlg
from alg.oe import OEScratch
import torch.nn.functional as F
import torchvision.transforms as trn
from dataset.tinyimages_80mn_loader import TinyImages
import torch
import numpy as np
from common.utils import AverageMeter, accuracy, ImbalancedDatasetSampler
import time
from torch.distributions import Categorical

class OODNoise(OEScratch):
    def __init__(self, args, gpu, num_classes, cls_num_list, train_dataset):
        super(OODNoise, self).__init__(args, gpu, num_classes, cls_num_list, train_dataset)

        self.ood_num = len(self.ood_data)
        cls_rate = np.array(cls_num_list)/np.array(cls_num_list).sum()
        print("The original class rate: ", cls_rate)
        if args.alpha > 0:
            self.rebalance_rate = torch.from_numpy(args.alpha - cls_rate).to(self.device)
        else:
            self.rebalance_rate = torch.from_numpy(cls_rate.max() + cls_rate.min() - cls_rate).to(
                self.device)
        print("Our rebalance rate: ", self.rebalance_rate)


    def train_batch_with_out(self, batch_idx, in_set, out_set, epoch):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]
        data, target = data.to(self.device), target.to(self.device)
        output = self.net(data)

        probs = self.rebalance_rate.reshape(1,-1).repeat(out_set[0].shape[0], 1)
        dist = Categorical(probs)
        target_random = dist.sample().reshape(out_set[0].shape[0])

        weight = self.rebalance_rate / self.rebalance_rate.sum() * self.num_classes

        loss = self.criterion(output[:in_set[0].shape[0]], target) + self.args.lambda_o * F.cross_entropy(
            output[in_set[0].shape[0]:], target_random, weight=weight.float())

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output[:len(in_set[0])], target, topk=(1, 5))

        return loss, acc1, acc5
