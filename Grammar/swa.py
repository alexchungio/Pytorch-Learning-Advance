#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File: swa.py
@Author: Alex Chung
@Time: 4/12/21 3:14 AM
@Concat: yonganzhong@outlook.com
"""


import os
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.tensorboard import SummaryWriter

import Grammar.utils as d2l


model_path = '../Outputs/swa/ckpt/model.pth'
swa_model_path = '../Outputs/swa/ckpt/swa_model.pth'
log_path = '../Outputs/swa/logs'

torch.manual_seed(2021)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
writer = SummaryWriter(log_dir=log_path)


def load_dataset(transforms=None):
    mnist_train = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST', train=True, download=True,
                                                    transform=transforms)
    mnist_test = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST', train=False, download=True,
                                                   transform=transforms)
    return mnist_train, mnist_test


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


class FashionModel(nn.Module):
    def __init__(self, num_classes):
        super(FashionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(num_features=64)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, train_loader, criterion, optimizer=None, log_interval=100):
    """

    Args:
        model:
        train_loader:
        criterion:
        optimizer:
        log_interval:

    Returns:

    """
    model.train()
    train_acc, train_loss = [], []
    num_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        train_acc.append((output.argmax(dim=1) == target).float().sum().item())
        num_samples += data.shape[0]
        if batch_idx % log_interval == 0:
            print('\t Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item()))

    # show train log
    train_acc = np.sum(train_acc) / num_samples
    train_loss = np.mean(train_loss)

    return train_acc, train_loss


def eval(model, eval_loader, criterion, log_interval=100):
    """

    Args:
        model:
        eval_loader:
        criterion:
        log_interval:

    Returns:

    """
    model.eval()

    eval_acc, eval_loss = [], []
    num_samples = 0

    for batch_idx, (data, target) in enumerate(eval_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)
        eval_loss.append(loss.item())
        eval_acc.append((output.argmax(dim=1) == target).float().sum().item())
        num_samples += data.shape[0]
        if batch_idx % log_interval == 0:
            print('\t Eval: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx * len(data), len(eval_loader.dataset),
                                                              100. * batch_idx / len(eval_loader), loss.item()))

    # show train log
    eval_acc = np.sum(eval_acc) / num_samples
    eval_loss = np.mean(eval_loss)

    return eval_acc, eval_loss


def save_model(net, model_path, complete_model=None):

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if complete_model is None:
        torch.save(net.state_dict(), model_path)
    else:
        torch.save(net, model_path)
    print('Successful save model...')


def load_model(net=None, model_path=None, complete_model=None):
    """

    Args:
        net:
        model_path:
        complete_model:

    Returns:

    """
    # x = torch.load(model_path)
    if complete_model is None:
        state_dict = torch.load(model_path)
        net.load_state_dict(state_dict)
        model = None
    else:
        model = torch.load(model_path)

    print('Successful load model...')
    return model


def main():

    # data
    mean = 0.1307
    std = 0.3081

    num_epochs = 120
    batch_size = 256
    num_workers = 4
    num_inputs = 28*28
    num_classes = 10
    lr = 0.1

    # swa
    swa_lr = 0.01
    swa_start = 100

    # ------------------------ dataset generator----------------------------------

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(mean,), std=(std,))
    ])

    mnist_train, mnist_eval = load_dataset(transform)

    # visualize
    # for x, Y in mnist_train:
    #     x = np.transpose(x, (1, 2, 0))
    #     plt.imshow(x)
    #     plt.show()
    #     break

    # dataloader
    train_generator = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_generator = data.DataLoader(mnist_eval, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    #---------------------------------model----------------------------------------

    model = FashionModel(num_classes=num_classes)
    model = model.to(device)

    # loss
    criterion = nn.CrossEntropyLoss()  # contain softmax operation

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_epochs, pct_start=.1, div_factor=10,
                                                    final_div_factor=10)

    # swa
    swa_model = AveragedModel(model=model)
    swa_scheduler = SWALR(optimizer, anneal_strategy="cos", anneal_epochs=5, swa_lr=swa_lr)

    for epoch in range(num_epochs):
        print('Epoch: {}'.format(epoch))
        train_acc, train_loss = train(model, train_generator, criterion, optimizer=optimizer)
        eval_acc, eval_loss = eval(model, eval_generator, criterion)
        writer.add_scalars('acc', {'train': train_acc,
                                   'eval': eval_acc}, global_step=epoch)
        writer.add_scalars('loss', {'train': train_loss,
                                   'eval': eval_loss}, global_step=epoch)

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)

        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            schedule.step()


    # save model
    save_model(model, model_path)

    # save swa model
    torch.optim.swa_utils.update_bn(train_generator, swa_model, device=device)

    swa_acc,  swa_loss = eval(swa_model, eval_generator, criterion)
    print('swa acc:{}, loss{}'.format(swa_acc, swa_loss))
    save_model(swa_model, swa_model_path)

    # inference
    # model = FashionModel(num_classes=num_classes)
    # load_model(model, model_path)
    # X, y = iter(eval_generator).next()
    # true_labels = get_fashion_mnist_labels(y.numpy())
    # pred_labels = get_fashion_mnist_labels(model(X).argmax(dim=1).numpy())
    # titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    # X = X * std + mean
    # show_fashion_mnist(X[0:9], titles[0:9])


if __name__ == "__main__":
    main()