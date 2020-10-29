#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tensorboard_cifar10.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/10/29 下午3:25
# @ Software   : PyCharm
#-------------------------------------------------------

import time
import torch
import torch.nn as nn
from visdom import Visdom
import torchvision.utils as vutils
import torchvision
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # visdom instance
writer = SummaryWriter(log_dir='../Outputs/Logs')


def load_dataset(batch_size, size=None, num_workers=4):
    # dataset process
    trans = []
    if size:
        trans.append(torchvision.transforms.Resize(size=size))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)

    # load
    mnist_train = torchvision.datasets.CIFAR10(root='../Datasets/CIFA10', train=True, download=True,
                                                    transform=transform)
    mnist_test = torchvision.datasets.CIFAR10(root='../Datasets/CIFA10', train=False, download=True,
                                                   transform=transform)
    # generate
    train_loader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, train_loader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 6,32,32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)

        self.relu1 = nn.ReLU()
        # 16,16,16
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # 64, 16, 16
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.relu2 = nn.ReLU()
        # 64，8, 8
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        # flatten
        self.fc1 = nn.Linear(in_features=256*4*4, out_features=128)
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.maxpool2(x)


        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout1(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout2(x)

        x = self.fc2(x)

        return x


def test(model, test_loader, loss, epoch, device=None):
    """

    """
    model.eval()  # convert to eval(model)

    if device is None and isinstance(model, torch.nn.Module):
        # if device is None, use the net device
        device = list(model.parameters())[0].device
    test_acc, test_loss, num_samples  = 0.0, 0.0, 0
    num_batch = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)  # load data to device
            y_pred = model(x)
            l = loss(y_pred, y)
            test_loss += l.cpu().item()
            test_acc += (y_pred.argmax(dim=1) == y).float().sum().cpu().item()
            num_samples += x.shape[0]
            num_batch += 1

        test_loss = test_loss / num_batch
        test_acc = test_acc / num_samples


        feature = x[0].unsqueeze(0)  # expend dimension (1, 32, 32) => (1, 1, 32, 32)
        image_grid = vutils.make_grid(feature[0], normalize=True, scale_each=True)
        writer.add_image('raw_image', image_grid, epoch)

        for name, layer in model._modules.items():

            if 'fc' in name:
                break
            else:
                feature = layer(feature)

            if 'conv' in name:
                # +++++++++++++++++++++++++(1,N,H,W) => (N,1,H,W)+++++++++++++++++++++++++++++
                feature_t = feature.transpose(0, 1)
                feature_grid = vutils.make_grid(F.relu(feature_t), normalize=True, scale_each=True, nrow=8)
                writer.add_image('{}_feature_map'.format(name), feature_grid, epoch)


        print('\teval => loss {:.4f}, acc {:.4f}'.
              format(test_loss, test_acc))
        return test_loss, test_acc


def train(model, train_loader, loss, optimizer, global_step, log_iter=None, device=None, ):

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

        acc = (pred_y.argmax(dim=1) == y).float().sum().cpu().item()
        train_acc += acc

        num_samples += x.shape[0]
        num_batch += 1

        if (global_step + 1) % log_iter == 0:
            writer.add_scalar(tag='train/loss', scalar_value=l.cpu().item(), global_step=global_step)
            writer.add_scalar(tag='train/acc', scalar_value= acc / x.shape[0], global_step=global_step)

            # add histogram of params
            for name, param in model.named_parameters():
                if 'bn' not in name:
                    writer.add_histogram(tag='param/{}'.format(name), values=param, global_step=global_step)

        global_step += 1


    train_loss = train_loss / num_batch
    train_acc = train_acc / num_samples


    print('\ttrain => loss {:.4f}, acc {:.4f}'.
          format(train_loss, train_acc))

    return train_loss, train_acc, global_step


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    num_epochs = 50
    batch_size = 256
    lr, gamma = 0.2, 0.9
    log_iter = 100
    model = CNN().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)  # SGDM

    # optimizer = optim.Adam(params=model.parameters(), lr=lr) # Adam
    scheduler = StepLR(optimizer, step_size=5, gamma=gamma)

    train_loader, test_loader = load_dataset(batch_size)

    global_step = 0
    for epoch in range(num_epochs):
        print('Epoch: {}:'.format(epoch + 1))
        train_loss, train_acc, global_step = train(model, train_loader, loss, optimizer, global_step, log_iter, device)
        test_loss, test_acc = test(model, test_loader, loss, epoch, device=device)
        scheduler.step(epoch)

        # add dict
        writer.add_scalars(main_tag='epoch/loss', tag_scalar_dict={'train':train_loss, 'val': test_loss}, global_step=epoch)
        writer.add_scalars(main_tag='epoch/acc', tag_scalar_dict={'train': train_acc, 'val': test_acc}, global_step=epoch)

    writer.add_graph(model, torch.randn(1, 3, 32, 32, device=device))
    torch.save(model.state_dict(), '../Outputs/cifar10/cifar10.pt')

    writer.close()



if __name__ == "__main__":
    main()