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

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # visdom instance
writer = SummaryWriter(log_dir='../Outputs/Logs/cifar10')

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)


def load_dataset(batch_size, num_workers=4):
    # dataset process

    test_trans = []

    train_trans = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    test_trans = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    train_transform = torchvision.transforms.Compose(train_trans)
    test_transform = torchvision.transforms.Compose(test_trans)

    # load
    cifar10_train = torchvision.datasets.CIFAR10(root='../Datasets/CIFA10', train=True, download=True,
                                                    transform=train_transform)
    cifar10_test = torchvision.datasets.CIFAR10(root='../Datasets/CIFA10', train=False, download=True,
                                                   transform=test_transform)
    # generate
    train_loader = data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def get_cifar10_labels(labels):
    text_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    return [text_labels[int(i)] for i in labels]



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


def plt_imshow(img):
    global mean
    global std
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array(mean)
    std = np.array(std)
    inp = std * img + mean
    img = np.clip(inp, 0, 1)
    plt.imshow(img)


def predict(model, images):
    """

    :param model:
    :param images:
    :return:  pred, prob
    """
    model.eval()
    output = model(images)
    # convert output probabilities to predicted class
    _, pred_tensor = torch.max(output, 1)
    pred = np.squeeze(pred_tensor.cpu().numpy())

    return pred, [F.softmax(el, dim=0)[i].item() for i, el in zip(pred, output)]


def plot_classes_predict(model, images, labels, device=None):

    if device is not None:
        preds, probs = predict(model, images.to(device))

    else:
        preds, probs = predict(model, images)

    classes = get_cifar10_labels(preds)

    fig = plt.figure(figsize=(64, 64))
    for idx in np.arange(100):
        ax = fig.add_subplot(10, 10, idx + 1, xticks=[], yticks=[])
        plt_imshow(images[idx])
        ax.set_title("{0}, {1:.1f} % \n (label: {2})".format(
            classes[idx],  # class
            probs[idx] * 100.0,  # prob
            labels[idx]), # labels
            color=("green" if preds[idx] == labels[idx].item() else "red"),
            fontsize = 28)
    return fig


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

        # write raw image to logs
        # feature * std + mean
        mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device).reshape((3, 1, 1))
        std_tensor = torch.tensor(std, dtype=torch.float32, device=device).reshape((3, 1, 1))
        image_tensor = feature[0] * std_tensor + mean_tensor
        image_grid = vutils.make_grid(image_tensor, normalize=False, scale_each=True)
        writer.add_image('raw_image', image_grid, epoch)

        # write model features to logs
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
    num_epochs = 120
    batch_size = 256
    lr, gamma = 0.1, 0.9
    log_iter = 100
    model = CNN().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)  # SGDM
    # optimizer = optim.ASGD(params=model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=40, gamma=gamma)

    train_loader, test_loader = load_dataset(batch_size)


    # dataset dimension visualize
    mnist_test = torchvision.datasets.MNIST(root='../Datasets/MNIST', train=False, download=True,
                                              transform=torchvision.transforms.ToTensor())
    images = mnist_test.data
    # images = np.transpose(images, (0,3,1,2))
    images = images[:,np.newaxis,:,:]  # (10000, 28, 28) => (10000, 1, 28, 28)
    images = images.clone().detach()
    labels = [mnist_test.classes[label].split('-')[1] for label in mnist_test.targets]
    features = images.view(-1, 1*28*28)

    writer.add_embedding(mat=features, metadata=labels, label_img=images, global_step=0)

    global_step = 0
    for epoch in range(num_epochs):
        print('Epoch: {}:'.format(epoch + 1))
        train_loss, train_acc, global_step = train(model, train_loader, loss, optimizer, global_step, log_iter, device)
        test_loss, test_acc = test(model, test_loader, loss, epoch, device=device)
        scheduler.step(epoch=epoch)

        # add loss and acc to logs
        writer.add_scalars(main_tag='epoch/loss', tag_scalar_dict={'train':train_loss, 'val': test_loss}, global_step=epoch)
        writer.add_scalars(main_tag='epoch/acc', tag_scalar_dict={'train': train_acc, 'val': test_acc}, global_step=epoch)

        # add learning_rate to logs
        writer.add_scalar(tag='lr', scalar_value=optimizer.param_groups[0]['lr'], global_step=epoch)

        images, labels = next(iter(test_loader))
        fig = plot_classes_predict(model, images, labels, device)
        writer.add_figure(tag='predict_vs_actual', figure=fig, global_step=epoch)

    # To save the model graph, the input_to_model should input a demo batch witch
    images_batch = torch.randn(1, 3, 32, 32, device=device)
    writer.add_graph(model, input_to_model=images_batch)

    model_path = '../Outputs/cifar10/cifar10.pt'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    writer.close()


if __name__ == "__main__":
    main()