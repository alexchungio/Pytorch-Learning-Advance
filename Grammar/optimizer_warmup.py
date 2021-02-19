#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : optimizer_warmup.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/2/19 上午11:35
# @ Software   : PyCharm
#-------------------------------------------------------
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler, StepLR, CosineAnnealingLR
from bisect import bisect_right
from transformers import get_linear_schedule_with_warmup

class WarmupMultiStepLR(_LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1 / 3,
            warmup_iters=2,
            warmup_method="linear",
            last_epoch=-1):
        """

        :param optimizer:
        :param milestones:
        :param gamma:
        :param warmup_factor:
        :param warmup_iters: (epoch)
        :param warmup_method:
        :param last_epoch:
        """

        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_ratio = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_ratio = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_ratio = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_ratio
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupCosineLR(_LRScheduler):
    def __init__(
            self,
            optimizer,
            num_training_step,
            warmup_factor=1 / 3,
            warmup_iters=2,
            warmup_method="linear",
            last_epoch=-1):
        """

        :param optimizer:
        :param num_training_step: (epoch)
        :param warmup_factor:
        :param warmup_iters: (epoch)
        :param warmup_method:
        :param last_epoch:
        """
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.num_training_step = num_training_step
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)


    def get_lr(self):
        warmup_ratio = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_ratio = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_ratio = self.warmup_factor * (1 - alpha) + alpha

            return [base_lr * warmup_ratio for base_lr in self.base_lrs]

        else:
            return [base_lr
                    * 0.5 * (math.cos((self.last_epoch - self.warmup_iters) /
                                      (self.num_training_step - self.warmup_iters) * math.pi)
                             + 1)
                    for base_lr in self.base_lrs]


def get_multi_step_schedule_with_warmup(optimizer, milestones, gamma, warmup_epochs):
    """

    :param optimizer:
    :param milestones:
    :param gamma:
    :param warmup_epochs:
    :return:
    """

    warm_up_with_multistep_lr = lambda \
            epoch: epoch / warmup_epochs if epoch <= warmup_epochs else gamma ** len(
        [m for m in milestones if m <= epoch])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)

    return scheduler


def get_cosine_schedule_with_warmup(optimizer, num_epochs, warmup_epochs):
    """

    :param optimizer:
    :param milestones:
    :param gamma:
    :param warmup_epochs:
    :return:
    """

    warm_up_with_cosine_lr = lambda epoch: epoch / warmup_epochs if epoch <= warmup_epochs else 0.5 * (
                math.cos((epoch - warmup_epochs) / (num_epochs - warmup_epochs) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    return scheduler


def main():
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.optim as optim
    from torchvision.models import resnet18

    epochs = 60
    base_lr = 0.01
    train_loader_size = 10
    warmup_epochs = 2
    warmup_factor = 1. / 3
    gamma = 0.1
    milestones = [40, 50]

    model = resnet18(num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, nesterov=True)

    # lr_scheduler = WarmupMultiStepLR(optimizer, [40, 70], warmup_iters=10)

    # scheduler = WarmupMultiStepLR(optimizer=optimizer,
    #                               milestones=milestones,
    #                               gamma=gamma,
    #                               warmup_factor=0.1,
    #                               warmup_iters=2,
    #                               warmup_method="linear")

    scheduler = WarmupCosineLR(optimizer=optimizer,
                               num_training_step=epochs,
                               warmup_factor=warmup_factor,
                               warmup_iters=warmup_epochs,
                               warmup_method="linear")
    lrs = []
    step = 1
    for epoch in range(epochs):
        for index, data in enumerate(np.arange(0, train_loader_size)):
            optimizer.zero_grad()
            optimizer.step()
            lrs.append(optimizer.param_groups[0]['lr'])
            print('step: {}'.format(step), 'lr: {}'.format(optimizer.param_groups[0]['lr']))
            scheduler.step(epoch + index / train_loader_size) # update lr every step
            step += 1
        # scheduler.step()  # update lr every epoch
    plt.plot(lrs, c='g', label='warmup step_lr', linewidth=1)
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    main()





