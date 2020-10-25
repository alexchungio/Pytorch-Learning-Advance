#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : ResNet.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/10/23 下午3:23
# @ Software   : PyCharm
#-------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_planes, output_planes, stride=1, norm_layer=None):
        super(BasicBlock, self).__init__()


        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # shortcut pass
        if stride != 1 or input_planes != output_planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(input_planes, output_planes * self.expansion, stride),
                norm_layer(output_planes * self.expansion)
            )
        else:
            self.downsample = None

        # main pass
        self.conv1 = conv3x3(input_planes, output_planes, stride)
        self.bn1 = norm_layer(output_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(output_planes, output_planes, stride=1)
        self.bn2 = norm_layer(output_planes)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_planes, output_planes, stride=1, norm_layer=None):
        super(Bottleneck, self).__init__()

        # main path
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # shortcut path
        if stride != 1 or input_planes != output_planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(input_planes, output_planes * self.expansion, stride),
                norm_layer(output_planes * self.expansion)
            )
        else:
            self.downsample = None

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(input_planes, output_planes)
        self.bn1 = norm_layer(output_planes)
        self.conv2 = conv3x3(output_planes, output_planes, stride)
        self.bn2 = norm_layer(output_planes)
        self.conv3 = conv1x1(output_planes, output_planes * self.expansion)
        self.bn3 = norm_layer(output_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):

        return x.view(x.shape[0], -1)


class ResNet18(nn.ModuleList):

    def __init__(self, num_classes):
        super(ResNet18, self).__init__()

        self.input_planes = 64
        # base block
        self.conv1 = nn.Conv2d(3, self.input_planes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # residual block

        self.res_block1 = nn.Sequential(
            BasicBlock(self.input_planes, 64, stride=1),
            BasicBlock(64, 64, stride=1)
        )

        self.res_block2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1)
        )

        self.res_block3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1)
        )

        self.res_block4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.flatten = Flatten()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)  # 64, 112, 112
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.maxpool(x)  # 64, 56, 56
        x = self.res_block1(x) # 64, 56, 56
        x = self.res_block2(x) # 128, 28, 28
        x = self.res_block3(x) # 256, 14, 14
        x = self.res_block4(x) # 512, 7, 7

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ResNet50(nn.ModuleList):

    def __init__(self, num_classes):
        super(ResNet50, self).__init__()

        self.input_planes = 64
        # base block
        self.conv1 = nn.Conv2d(3, self.input_planes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # residual block

        self.res_block1 = nn.Sequential(
            Bottleneck(self.input_planes, 64, stride=1),
            Bottleneck(256, 64, stride=1),
            Bottleneck(256, 64, stride=1),
            Bottleneck(256, 64, stride=1)
        )

        self.res_block2 = nn.Sequential(
            Bottleneck(256, 128, stride=2),
            Bottleneck(512, 128, stride=1),
            Bottleneck(512, 128, stride=1),
            Bottleneck(512, 128, stride=1)
        )

        self.res_block3 = nn.Sequential(
            Bottleneck(512, 256, stride=2),
            Bottleneck(1024, 256, stride=1),
            Bottleneck(1024, 256, stride=1),
            Bottleneck(1024, 256, stride=1),
            Bottleneck(1024, 256, stride=1),
            Bottleneck(1024, 256, stride=1)
        )

        self.res_block4 = nn.Sequential(
            Bottleneck(1024, 512, stride=2),
            Bottleneck(2048, 512, stride=1),
            Bottleneck(2048, 512, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)  # 64, 56, 56

        x = self.res_block1(x) # 256, 56, 56
        x = self.res_block2(x) # 512, 28, 28
        x = self.res_block3(x) # 1024, 14, 14
        x = self.res_block4(x) # 2048, 7, 7

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def main():
    x = torch.randn((10, 3, 224, 224))

    resnet_18 = ResNet18(10)

    resnet_50 = ResNet50(10)
    y = resnet_50(x)

    print(y.size())


if __name__ == "__main__":
    main()








