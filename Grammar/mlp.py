#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : mlp.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/10/19 下午4:52
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
import torch.nn.functional as F

import Grammar.utils as d2l


# Module ModuleList Sequential
# ModuleList  container
# Sequential  order container


model_path = '../Outputs/mlp/mnist.pt'

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


def save_model(model, model_path, complete_model=None):

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
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
        torch.save(model.state_dict(), model_path)
    else:
        torch.save(model, model_path)
    print('Successful save model...')



def load_model(model=None, model_path=None, complete_model=None):

    # x = torch.load(model_path)


    if complete_model is None:
        model.load_state_dict(torch.load(model_path))
        model = None
    else:
        model = torch.load(model_path)

    print('Successful load model...')
    return model


class MLP(nn.Module):
    def __init__(self, num_dim):
        super(MLP, self).__init__()
        self.flatten = Flatten()
        self.linears = nn.ModuleList([nn.Linear(inp, out) for inp, out in zip(num_dim[:-1], num_dim[1:])])
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        # self.relu1 = nn.ReLU()
        # self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linears[0](x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linears[1](x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linears[2](x)
        return x

def main(mode):

    num_epochs = 10
    batch_size = 128
    num_workers = 4

    num_inputs = 28 * 28
    num_hidden1 = 256
    num_hidden2 = 126
    num_outputs = 10
    lr = 0.2

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # ------------------------ dataset generator----------------------------------
    mnist_train, mnist_test = load_dataset()

    # X, y = [], []
    # for i in range(10):
    #     X.append(mnist_train[i][0])
    #     y.append(mnist_train[i][1])
    # show_fashion_mnist(X, get_fashion_mnist_labels(y))
    train_generator = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_generator = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # net = nn.Sequential(OrderedDict(
    #     [('flatten', Flatten()),
    #      ('linear_1', nn.Linear(num_inputs, num_hiddens)),
    #      ('linear_2', nn.Linear(num_hiddens, num_outputs))
    #      ])
    # )

    model = MLP([num_inputs, num_hidden1, num_hidden2, num_outputs])
    # print(net)
    # initial parameter
    for param in model.parameters():
        nn.init.normal_(param, mean=0, std=0.01)

    if mode == 'train':
        # **********************************************************
        # change model mode to train => drop_prob = drop_prob
        # **********************************************************
        model.train()
        loss = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        train(model, train_generator, test_generator, loss, num_epochs, batch_size, optimizer=optimizer)

        # save model
        save_model(model, model_path)

    elif mode == 'test':
        # **********************************************************
        # change model mode to eval => drop_prob = 0.0
        # **********************************************************
        model.eval()
        load_model(model, model_path)

        x, y = iter(test_generator).next()

        true_labels = get_fashion_mnist_labels(y.numpy())
        pred_labels = get_fashion_mnist_labels(model(x).argmax(dim=1).numpy())
        titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

        show_fashion_mnist(x[0:9], titles[0:9])

if __name__ == "__main__":
    main('train')
    main('test')
