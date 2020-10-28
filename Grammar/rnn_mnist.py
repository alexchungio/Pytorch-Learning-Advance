#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : rnn_mnist.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/10/26 下午2:52
# @ Software   : PyCharm
#-------------------------------------------------------

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from IPython import display
import numpy as np



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_dataset(size=None):

    trans = []
    if size :
        trans.append(transforms.Resize(size))

    trans.append(transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)

    mnist_train = torchvision.datasets.MNIST(root='../Datasets/MNIST', train=True, download=True,
                                                    transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='../Datasets/MNIST', train=False, download=True,
                                                   transform=transform)

    return mnist_train, mnist_test


def get_mnist_labels(labels):
    text_labels = ['zero', 'one', 'two', 'three', 'four',
                   'five', 'six', 'seven', 'eight', 'nine']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    display.set_matplotlib_formats('svg')
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def init_orthogonal(param):
    """
    Initializes weight parameters orthogonally.
    This is a common initiailization for recurrent neural networks.

    Refer to this paper for an explanation of this initialization:
    https://arxiv.org/abs/1312.6120
    """
    if param.ndim < 2:
        raise ValueError("Only parameters with 2 or more dimensions are supported.")

    rows, cols = param.shape

    new_param = np.random.randn(rows, cols)

    if rows < cols:
        new_param = new_param.T

    # Compute QR factorization
    q, r = np.linalg.qr(new_param)

    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = np.diag(r, 0)
    ph = np.sign(d)
    q *= ph

    if rows < cols:
        q = q.T

    new_param = q

    return new_param


def sigmoid(x, derivative=False):
    """
    Computes the element-wise sigmoid activation function for an array x.

    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-12
    f = 1 / (1 + np.exp(-x_safe))

    if derivative:  # Return the derivative of the function evaluated at x
        return f * (1 - f)
    else:  # Return the forward pass of the function at x
        return f


def tanh(x, derivative=False):
    """
    Computes the element-wise tanh activation function for an array x.

    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-12
    f = (np.exp(x_safe) - np.exp(-x_safe)) / (np.exp(x_safe) + np.exp(-x_safe))

    if derivative:  # Return the derivative of the function evaluated at x
        return 1 - f ** 2
    else:  # Return the forward pass of the function at x
        return f


def softmax(x, derivative=False):
    """
    Computes the softmax for an array x.

    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-12
    f = np.exp(x_safe) / np.sum(np.exp(x_safe))

    if derivative:  # Return the derivative of the function evaluated at x
        pass  # We will not need this one
    else:  # Return the forward pass of the function at x
        return f

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden and cell states
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)

        # Forward propagate LSTM
        out, _ = self.rnn(x, h_0)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out


def test(model, test_loader, sequence_length, input_size, epoch, device):

    model.eval()
    with torch.no_grad():
        test_acc, num_samples = 0, 0

        for images, labels in test_loader:
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)

            pred_y = model(images)
            test_acc += (pred_y.argmax(dim=1) == labels).float().sum().cpu().item()
            num_samples += images.shape[0]

        print('Test epoch {} => acc {:.4f}'.
              format(epoch, test_acc / num_samples))

def train(model, train_loader, sequence_length, input_size, loss, optimizer, epoch, device):
    """

    :param model:
    :param train_loader:
    :param sequence_length:
    :param input_size:
    :param device:
    :param loss:
    :param optimizer:
    :param num_epochs:
    :return:
    """
    model.train()

    train_acc, train_loss, num_samples = 0, 0.0, 0
    num_batch = 0

    for images, labels in train_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        pred_y = model(images)
        l = loss(pred_y, labels)
        # grad clearing
        optimizer.zero_grad()
        # computer grad
        l.backward()
        # update grad
        optimizer.step()

        train_loss += l.cpu().item()
        train_acc += (pred_y.argmax(dim=1) == labels).float().sum().cpu().item()

        num_samples += images.shape[0]
        num_batch += 1

    print('Train epoch {} => loss {:.4f}, acc {:.4f}'.
          format(epoch, train_loss / num_batch, train_acc / num_samples))

def main():

    mnist_train, mnist_test = load_dataset()

    num_epochs = 10
    batch_size = 256
    num_workers = 4
    lr = 0.001
    num_layers = 2
    input_size = 28
    sequence_length = 28
    hidden_size = 128
    output_size = 10

    model = RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    train_loader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    for epoch in range(num_epochs):
        train(model, train_loader, sequence_length, input_size, loss=loss, optimizer=optimizer, device=device,
              epoch=epoch+1)
        test(model, test_loader, sequence_length, input_size, epoch+1, device)



if __name__ == "__main__":
    main()



