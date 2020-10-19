#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : regression.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/10/18 下午4:03
# @ Software   : PyCharm
#-------------------------------------------------------

import torch
from torch import nn
from torch import optim
import numpy as np
import torch.utils.data as Data


# config
batch_size = 100
lr = 0.02
num_epochs = 10

num_inputs = 2
num_examples = 1000

# generate dataset
t_w = [2., -3.]
t_b = 1.
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = t_w[0] * features[:, 0] + t_w[1] * features[:, 1] + t_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)


def generate_data(features, labels, batch_size):
    """

    :param features:
    :param labels:
    :param batch_size:
    :return:
    """
    # combine feature and label
    dataset = Data.TensorDataset(features, labels)
    # random read batch dataset
    data_generate = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return data_generate


# define model
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)  # use autograd to define model

    def forward(self, x):
        y = self.linear(x)

        return y


def main():
    # for x, y in generate_data(features, labels, batch_size):
    #     print(x, y)
    #     break
    net = LinearNet(num_inputs)

    # print(net)

    # show parameter
    # for param in net.parameters():
    #     print(param)

    # init parameter
    nn.init.normal_(net.linear.weight, mean=0, std=0.01)
    nn.init.constant_(net.linear.bias, val=0)


    # define loss function
    loss = nn.MSELoss()

    # define optimizer
    optimizer = optim.SGD(params=[{'params': net.parameters()}], lr=lr)

    # print(optimizer)

    for epoch in range(num_epochs):
        for x, y in generate_data(features, labels, batch_size):
            outputs = net(x)
            l = loss(outputs, y.view(outputs.size()))
            optimizer.zero_grad() # equal to net.zero_grad()
            l.backward()
            optimizer.step()

        print('epoch {} => loss {:.4}'.format(epoch + 1, l.item()))


    print(t_w, '<=>', net.linear.weight.data)
    print(t_b, '<=>', net.linear.bias.data)


if __name__ == "__main__":
    main()