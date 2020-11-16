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
import torch.utils.data as data

import pedestrian_detection.data.transforms as T
from pedestrian_detection.libs import utils


parser = argparse.ArgumentParser(description='Finetuning a pre-trained Mask R-CNN model in the Penn-Fudan Database for '
                                             'Pedestrian Detection and Segmentation.')
parser.add_argument('--dataset', default='/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/PennFudanPed', type=str, help='dataset path')

args = parser.parse_args()

torch.manual_seed(20201116)

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

    num_pedestrian = len(np.unique(np.array(mask))[1:]) # remove background
    print('Number pedestrian {}'.format(num_pedestrian))

    color_mode = [0, 0, 255, # black background
                  255, 0, 0, # index 1 is red
                  255, 255, 0, # index 2 is yellow
                  255, 153, 0, # index 3 is orange
                  ]
    mask.putpalette(color_mode)
    mask.show()
    print('Done')


class PennFudanDataset(data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of binary masks (background | pedestrian)
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transforms(is_training=False):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if is_training:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    visual_dataset()
    dataset_train = PennFudanDataset(args.dataset, transforms=get_transforms(is_training=True))
    dataset_test = PennFudanDataset(args.dataset, transforms=get_transforms(is_training=False))


    # split dataset to train and test
    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])


    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True, num_workers=4,collate_fn=utils.collate_fn)

    test_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    images, target = next(iter(train_loader))

    print('Done')

if __name__ == "__main__":
    main()


