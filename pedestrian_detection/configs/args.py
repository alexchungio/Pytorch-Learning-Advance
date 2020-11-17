#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : args.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/17 上午9:28
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description='Finetuning a pre-trained Mask R-CNN model in the Penn-Fudan Database for '
                                             'Pedestrian Detection and Segmentation.')
parser.add_argument('--dataset', default='/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/PennFudanPed', type=str, help='dataset path')
parser.add_argument('--checkpoint', default=os.path.join(ROOT, 'outputs', 'checkpoint', 'model.pth'), type=str,
                    help='checkpoint save path')

args = parser.parse_args()


if __name__ == "__main__":
    print(args.checkpoint)