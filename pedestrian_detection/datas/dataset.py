#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : dataset.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/16 下午1:49
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
from PIL import Image
import argparse

import torch



parser = argparse.ArgumentParser(description='Finetuning a pre-trained Mask R-CNN model in the Penn-Fudan Database for '
                                             'Pedestrian Detection and Segmentation.')
parser.add_argument('--dataset', default='/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/PennFudanPed', type=str, help='dataset path')

args = parser.parse_args()


def visual_dataset():
    # show rgb image
    image_path = os.path.join(args.dataset, 'PNGImages', 'FudanPed00001.png')
    image = Image.open(image_path)
    print('RGB image info: size {} mode {}'.format(image.size, image.mode))
    image.show()

    # show image mask
    mask_path = os.path.join(args.dataset, 'PedMasks', 'FudanPed00001_mask.png')
    mask = Image.open(mask_path)
    print('Mask image info: size {} mode {}'.format(mask.size, mask.mode))

    num_pedestrian = len(set(np.array(mask).flatten().tolist())) - 1  # remove background
    print('Number pedestrian {}'.format(num_pedestrian))

    color_mode = [0, 0, 255, # black background
                  255, 0, 0, # index 1 is red
                  255, 255, 0, # index 2 is yellow
                  255, 153, 0, # index 3 is orange
                  ]
    mask.putpalette(color_mode)
    mask.show()
    print('Done')


def main():
    visual_dataset()

if __name__ == "__main__":
    main()


