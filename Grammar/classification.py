#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : classification.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/10/19 上午9:28
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import OrderedDict
import Torch_Grammar.utils as d2l


# Module ModuleList Sequential
# ModuleList  container
# Sequential  order container


model_path = '../outputs/mnist.pt'

def load_dataset():
    mnist_train = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST', train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST', train=False, download=True,
                                                   transform=transforms.ToTensor())
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


# class LinearNet(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(LinearNet, self).__init__()
#         self.linear = nn.Linear(num_inputs, num_outputs)
#
#     def forward(self, x):
#         y = self.linear(input=x.view(x.shape[0], 1))
#         return y

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):

        return x.view(x.shape[0], -1)


def evaluate_accuracy(data_generator, net):
    acc_sum, num = 0, 0
    for x, y in data_generator:
        pred = net(x)
        acc_sum += (pred.argmax(dim=1) == y).float().sum().item()
        num += x.shape[0]
    return acc_sum / num


def train(net, train_generator, test_generator, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_acc_sum, train_num = 0., 0
        train_loss_sum = 0.0
        for x, y in train_generator:
            y_pred = net(x)

            # computer loss
            l = loss(y_pred, y).sum()

            if optimizer is not None:
                optimizer.zero_grad()
            # clean zero grad
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            # compute grad
            l.backward()

            # update parameter
            optimizer.step()

            train_loss_sum += l.item()
            train_acc_sum += (y_pred.argmax(dim=1) == y).float().sum().item()
            train_num += x.shape[0]

        # show train log
        train_acc = train_acc_sum / train_num
        train_loss = train_loss_sum / train_num

        test_acc = evaluate_accuracy(test_generator, net)

        print('epoch {} => loss {:.4f}, train acc {:.4f}, test acc {:.4f}'.
              format(epoch + 1, train_loss, train_acc, test_acc))


def save_model(net, model_path, complete_model=None):

    # os.makedirs(os.path.dirname(model_path), exist_ok=True)
    #
    # x = torch.tensor(3, dtype=torch.float)
    # y = torch.tensor(4, dtype=torch.float)
    #
    # xy_list = [x, y]
    # xy_dict = {'x': x, 'y': y}
    # # torch.save(x, model_path)
    # # torch.save(xy_list, model_path)
    # torch.save(xy_dict,  model_path)
    if complete_model is None:
        torch.save(net.state_dict(), model_path)
    else:
        torch.save(net, model_path)
    print('Successful save model...')



def load_model(net=None, model_path=None, complete_model=None):

    # x = torch.load(model_path)


    if complete_model is None:
        net.load_state_dict(torch.load(model_path))
        model = None
    else:
        model = torch.load(model_path)

    print('Successful load model...')
    return model


def main(mode):

    num_epochs = 10
    batch_size = 128
    num_workers = 4

    num_inputs = 28*28
    num_outputs = 10
    lr = 0.1



    # ------------------------ dataset generator----------------------------------
    mnist_train, mnist_test = load_dataset()

    # X, y = [], []
    # for i in range(10):
    #     X.append(mnist_train[i][0])
    #     y.append(mnist_train[i][1])
    # show_fashion_mnist(X, get_fashion_mnist_labels(y))
    train_generator = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_generator = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    #---------------------------------model----------------------------------------

    net = nn.Sequential(OrderedDict(
        [('flatten', Flatten()),
         ('linear', nn.Linear(num_inputs, num_outputs))
         ])
    )

    if mode == 'train':

        # init weight and bias
        nn.init.normal_(net.linear.weight, mean=0., std=0.01)
        nn.init.constant_(net.linear.bias, val=0.)

        # loss
        loss = nn.CrossEntropyLoss()

        # optimizer
        optimizer = optim.SGD(net.parameters(), lr=lr)

        train(net, train_generator, test_generator, loss, num_epochs, batch_size, optimizer=optimizer)

        # save model
        save_model(net, model_path)
    elif mode == 'test':
        load_model(net, model_path)

        X, y = iter(test_generator).next()

        true_labels = get_fashion_mnist_labels(y.numpy())
        pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
        titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

        show_fashion_mnist(X[0:9], titles[0:9])

if __name__ == "__main__":
    # main(mode='train')
    main(mode='test')

