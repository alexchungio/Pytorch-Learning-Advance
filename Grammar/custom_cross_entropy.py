#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : custom_bce_ce_loss.py
# @ Description:
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/26 下午5:12
# @ Software   : PyCharm
#-------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(input, target, weight=None, use_logit=True):
    if use_logit:
        input = torch.softmax(input, dim=1)
        input = torch.log(input)

    target = F.one_hot(target)

    output = - target * input

    if weight is not None:
        new_size = (1, weight.size(0)) if len(weight.size()) else weight.size()
        weight = weight.expand(new_size)

        # note: output * weight equal to execute mean at 1 dimension
        # and there require only execute sum at dimension 1
        # step 1: normalize weight
        weight =  weight / weight.sum()
        # step 2: recover weight to true scale
        weight *= weight.size(1)
        # step 3: apply weight
        output = output * weight

    output = output.sum(dim=1) # add class weight

    return torch.mean(output)


def binary_cross_entropy_loss(input, target, weight=None, use_logit=False):
    """
    -(y_n * ln(x_n) + (1 - y_n) * ln(1-x_n))
    :param input:
    :param output:
    :param use_logit:
    :return:
    """
    if use_logit:
        input = torch.sigmoid(input)

    output = - ((target * torch.log(input)) + (1. - target) * torch.log(1. - input))

    if weight is not None:
        new_size = (1, weight.size(0)) if len(weight.size()) else weight.size()
        weight = weight.expand(new_size)
        # step 2: apply weight
        output = output * weight

    return torch.mean(output)



def main():
    torch.random.manual_seed(2020)

    input = torch.randn(2, 2)
    weight = torch.tensor([0.4, 0.7], dtype=torch.float32)
    multi_label_target = torch.tensor([[1, 0],
                                       [1, 1]], dtype=torch.float32)
    multi_class_target = torch.tensor([0, 1], dtype=torch.long)


    # test nn.BCELoss
    m_0 = nn.Sigmoid()

    bce_criterion = nn.BCELoss(weight=weight)
    output_0 = bce_criterion(m_0(input), multi_label_target)
    output_1 = binary_cross_entropy_loss(input, multi_label_target, weight=weight, use_logit=True)
    print(output_0)
    print(output_1)

    # test nn.CrossEntropyLoss
    ce_criterion = nn.CrossEntropyLoss(weight=weight, reduction='mean')
    output_2 = ce_criterion(input, multi_class_target)
    output_3 = cross_entropy_loss(input, multi_class_target, weight=weight, use_logit=True)
    print(output_2)
    print(output_3)

    # test nn.NLLLoss
    logit_input = torch.log(F.softmax(input, dim=1))
    nll_criterion = nn.NLLLoss(weight=weight, reduction='mean')
    output_4 = ce_criterion(logit_input, multi_class_target)

    output_5 = cross_entropy_loss(logit_input, multi_class_target, weight=weight, use_logit=False)
    print(output_4)
    print(output_5)


if __name__ == "__main__":
    main()
