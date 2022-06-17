from alg.common.base_framework import BaseAlg
from common.utils import accuracy


class Standard(BaseAlg):

    def train_batch(self, batch_idx, input, target, epoch):
        if self.args.gpu is not None:
            input = input.to(self.device)
        target = target.to(self.device)

        # compute output
        output = self.net(input)
        loss = self.criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        return loss, acc1, acc5