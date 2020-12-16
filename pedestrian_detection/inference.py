#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inference.py
# @ Description:  https://www.rapidtables.com/web/color/RGB_Color.html
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/17 上午10:52
# @ Software   : PyCharm
#-------------------------------------------------------


import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from pedestrian_detection.configs.args import args
from pedestrian_detection.model.mask_rcnn import *
from pedestrian_detection.data.dataset import PennFudanDataset, get_transform


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def visual_mask(masks, mask_threshold=0.5):
    """
    show predict result as mask image
    :param masks: [N, 1, H, W] probability maps between 0-1. where N is the number of predictions
    :param mask_threshold:
    :return:
    """
    # reference https://www.rapidtables.com/web/color/RGB_Color.html
    color_mode = [0, 0, 255,  # black background
                  255, 0, 0,  # red
                  255, 255, 0,  # yellow
                  255, 153, 0,  # range
                  0, 255, 255,
                  153, 153, 255,
                  ]
    mask = masks.squeeze(dim=1)

    # filter pixel to 0 or 1
    # one_mask = torch.ones_like(mask, device=device)
    # zero_mask = torch.zeros_like(mask, device=device)
    # mask = torch.where(mask >= mask_threshold, one_mask, mask)
    # mask = torch.where(mask < mask_threshold, zero_mask, mask)
    mask = mask > mask_threshold
    mask = mask.to(dtype=torch.int32)

    # assign label to instance mask
    mask_label = torch.range(1, mask.size()[0], device=device).view(mask.size()[0], 1, 1)
    mask = mask * mask_label
    # add instance mask
    mask = mask.sum(dim=0, dtype=torch.int32)

    # convert tensor to Image
    mask_image = Image.fromarray(mask.cpu().numpy(), mode='I')
    mask_image = mask_image.convert(mode='L')

    mask_image.putpalette(color_mode)

    return mask_image


def inference(model, image, mask_threshold=0.5, device=None):

    model.eval()

    model.to(device)
    with torch.no_grad():
        prediction = model([image.to(device)])

    # show raw image
    # (0, 1) => (0, 255)
    # (C, H, W) => (H, W, C)
    rgb_image = Image.fromarray(image.mul(255).permute(1, 2, 0).byte().numpy())
    # rgb_image.show()

    # show predict result as mask image
    masks = prediction[0]['masks']
    mask_image = visual_mask(masks, mask_threshold)
    # mask_image.show()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8))

    ax0.set_title('raw image')
    ax0.imshow(np.asarray(rgb_image))
    ax0.axis('off')
    ax1.set_title('mask image')
    ax1.imshow(np.asarray(mask_image))
    ax1.axis('off')

    plt.show()



def main():
    dataset_test = PennFudanDataset(args.dataset, transforms=get_transform(is_training=False))

    image, _ = dataset_test[0]

    # define model
    model = mask_rcnn(num_classes=2)
    # load checkpoint
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict)

    inference(model, image, device=device)

if __name__ == "__main__":
    main()

    a = torch.range(1, 2)

