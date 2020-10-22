#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : cat_stack.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/10/22 上午9:43
# @ Software   : PyCharm
#-------------------------------------------------------

import torch



def main():
    a0 = torch.randn(28, 28, 1)
    a1 = torch.randn(28, 28, 1)

    b0 = torch.randn(28, 28, 2)
    b1 = torch.randn(28, 28, 2)

    # cat 在对应维度对 tensor 进行拼接
    # 输入的 tensor，必须保证除了拼接维度之外的其他维度具有相同的形状
    # 输出的维度大小不变
    c0 = torch.cat((a0, a1), dim=2)  # concat
    print(c0.shape)  # torch.Size([28, 28, 2])
    c1 = torch.cat((a0, b0), dim=2)
    print(c1.shape)  # torch.Size([28, 28, 3])

    # stack 在对应维度对两个 tensor 进行对叠
    # 输入的 tensor 在所有维度都具有相同的形状

    # s0 = torch.stack((a0, b0), dim=2)  # error
    s0 = torch.stack((a0, a1), dim=2)
    print(s0.shape)  # torch.Size([28, 28, 2, 2])
    s1 = torch.stack((b0, b1), dim=2)
    print(s1.shape)  # torch.Size([28, 28, 2, 2])



if __name__ == "__main__":
    main()
