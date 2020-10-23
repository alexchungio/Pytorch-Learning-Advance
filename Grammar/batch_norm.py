#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : batch_norm.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/10/22 下午4:01
# @ Software   : PyCharm
#-------------------------------------------------------
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import Grammar.utils as d2l


def load_dataset(batch_size, size=None, num_workers=4):
    # dataset process
    trans = []
    if size:
        trans.append(torchvision.transforms.Resize(size=size))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)

    # load
    mnist_train = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST', train=True, download=True,
                                                    transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST', train=False, download=True,
                                                   transform=transform)
    # generate
    train_generator = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_generator = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_generator, test_generator


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # Conv2d: in_channels, out_channels, kernel_size, stride=1, padding=0
        # 1,32,32
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # 6,28 ,28
        self.batch_norm1 = nn.BatchNorm2d(num_features=6)
        self.sigmoid1 = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool2d(2, 2)  # 6,14,14

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 16,10,10
        self.batch_norm2 = nn.BatchNorm2d(num_features=16)
        self.sigmoid2 = nn.Sigmoid()
        self.maxpool2 = nn.MaxPool2d(2, 2)  # 16,5,5

        # flatten 16*5*5

        # Linear: in_features, out_features, bias=True
        # fc1
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.batch_norm3 = nn.BatchNorm1d(num_features=120)
        self.sigmoid3 = nn.Sigmoid()

        # fc2
        self.fc2 = nn.Linear(120, 84)
        self.batch_norm4 = nn.BatchNorm1d(num_features=84)
        self.sigmoid4 = nn.Sigmoid()

        # fc3
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.sigmoid1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.sigmoid2(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.batch_norm3(x)
        x = self.sigmoid3(x)

        x = self.fc2(x)
        x = self.batch_norm4(x)
        x = self.sigmoid4(x)

        x = self.fc3(x)

        return x


def test(model, test_loader, epoch, device=None):
    """

    """
    model.eval()  # convert to eval(model)

    if device is None and isinstance(model, torch.nn.Module):
        # if device is None, use the net device
        device = list(model.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)  # load data to device
            acc_sum += (model(x).argmax(dim=1) == y).float().sum().cpu().item()
            n += x.shape[0]

    print('Eval epoch {} => acc {:.4f}'.format(epoch, acc_sum / n))


def train(model, train_loader, loss, optimizer, epoch, device=None):
    """
    convert train model
    """
    model.train()

    train_acc, train_loss, num_samples = 0, 0.0, 0
    num_batch = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred_y = model(x)
        l = loss(pred_y, y)
        # grad clearing
        optimizer.zero_grad()
        # computer grad
        l.backward()
        # update grad
        optimizer.step()

        train_loss += l.cpu().item()
        train_acc += (pred_y.argmax(dim=1) == y).float().sum().cpu().item()

        num_samples += x.shape[0]
        num_batch += 1

    print('Train epoch {} => loss {:.4f}, acc {:.4f}'.
          format(epoch, train_loss / num_batch, train_acc / num_samples))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    num_epochs = 20
    batch_size = 256
    lr, gamma = 0.1, 0.9
    model = LeNet().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)  # SGDM

    # optimizer = optim.Adam(params=model.parameters(), lr=lr) # Adam
    scheduler = StepLR(optimizer, step_size=2, gamma=gamma)

    train_loader, test_loader = load_dataset(batch_size, size=(32, 32))

    for epoch in range(num_epochs):
        train(model, train_loader, loss, optimizer, epoch + 1, device)
        test(model, test_loader, epoch + 1, device=device)
        scheduler.step(epoch)

if __name__ == "__main__":
    main()