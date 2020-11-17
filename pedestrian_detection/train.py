#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/16 下午5:04
# @ Software   : PyCharm
#-------------------------------------------------------
import os
import torch
import torchvision
from skimage.draw import polygon2mask


from pedestrian_detection.model.faster_rcnn import *
from pedestrian_detection.model.mask_rcnn import *
from pedestrian_detection.libs import utils
from pedestrian_detection.data.dataset import PennFudanDataset, get_transform
from pedestrian_detection.configs.args import args
from pedestrian_detection.libs.engine import train_one_epoch, evaluate


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(model, optimizer, criterion, ):
    pass


def eval():
    pass



def model_test():

    model = mask_rcnn(num_classes=2, pretrained=False)
    dataset = PennFudanDataset(args.dataset, get_transform(is_training=True))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    # For Training
    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]  # convert tuple to list
    model.train()
    output = model(images, targets)  # Returns losses and detections
    print(output)
    # For inference
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)  # Returns predictions
    print(predictions)


def main():
    # model_test()

    num_classes = 2
    num_epochs = 10
    lr = 0.005

    dataset_train = PennFudanDataset(args.dataset, transforms=get_transform(is_training=True))
    dataset_test = PennFudanDataset(args.dataset, transforms=get_transform(is_training=False))

    # split dataset to train and test
    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

    test_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)


    # get the model using our helper function
    model = mask_rcnn(num_classes, pretrained=True)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step(epoch=epoch)
        # evaluate on the test dataset
        evaluate(model, data_loader=test_loader,  device=device)

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)

if __name__ == "__main__":
    main()