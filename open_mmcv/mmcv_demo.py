#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : mmcv_demo.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/4 下午2:49
# @ Software   : PyCharm
#-------------------------------------------------------

import torch
import torchvision
import mmcv
import numpy as np
import matplotlib.pyplot as plt

image_path = '../Dataset/sunflower.jpg'


def main():

    bgr_img = mmcv.imread(image_path)

    h, w, _ = bgr_img.shape
    # convert color
    rgb_img = mmcv.bgr2rgb(bgr_img)

    # resize
    resize_img = mmcv.imresize(rgb_img, size=(256, 256))

    # rotate
    rotate_img =mmcv.imrotate(rgb_img, angle=45)

    # flip
    flip_img = mmcv.imflip(rgb_img, direction='horizontal')

    # crop
    if h <= w:
        y_min, y_max = 0, h
        x_min = int(( w - h) / 2)
        x_max = x_min + h
    else:
        x_min, x_max = 0, h
        y_min = int((h - w) / 2)
        y_max = y_min + w
    bbox = np.array([x_min, y_min, x_max, y_max])
    crop_img = mmcv.imcrop(rgb_img, bbox)

    # padding
    max_size = max(h, w)
    pad_img = mmcv.impad(rgb_img, shape=(max_size, max_size), padding_mode='constant')


    plt.imshow(pad_img)
    plt.axis('off')
    plt.show()



if __name__ == "__main__":
    main()