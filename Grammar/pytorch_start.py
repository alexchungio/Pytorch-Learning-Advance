#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : pytorch_start.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/1/21 PM 4:18
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import torch
import cv2 as cv

if __name__ == "__main__":
    # a = torch.rand(5, 3)
    # b = torch.tensor([[1, 3], [2, 4]])
    # print(b)

    print(torch.cuda.is_available())

    a = torch.rand((3, 3)).cuda()
    b = a.to(device='cpu', dtype=torch.double)
    print(a)
    print(b)


    torch.gather()











